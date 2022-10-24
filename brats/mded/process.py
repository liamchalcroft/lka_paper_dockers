from weakref import ref
import SimpleITK
import numpy as np
import json
import os
from pathlib import Path

DEFAULT_INPUT_PATH = Path("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/tumour-segmentation/results.json")

import torch
from nnunet.nn_unet import NNUnet
from types import SimpleNamespace
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from tqdm import tqdm
from data_preprocessing.preprocessor import Preprocessor
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary, RichProgressBar
from utils.utils import verify_ckpt_path
from data_loading.data_module import DataModule
from copy import deepcopy
import shutil
from skimage.morphology import remove_small_objects, remove_small_holes
import pathlib

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# todo change with your team-name
class ploras:
    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH,
    ):

        self.debug = True  # False for running the docker!
        if self.debug:
            self._input_path = Path(
                "/home/lchalcroft/mdunet/data/BraTS2021_val/images"
            )
            self._output_path = Path(
                "/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/tumour-segmentation"
            )
            self._algorithm_output_path = (
                self._output_path / "tumour-segmentation"
            )
            self._output_file = self._output_path / "results.json"
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = (
                self._output_path / "stroke-lesion-segmentation"
            )
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        tta = True

        args = SimpleNamespace(
            exec_mode="predict",
            data="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/data/11_3d/test",
            results="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/results",
            config="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/configs/config.pkl",
            logname="ploras",
            task="11",
            gpus=1,
            nodes=1,
            learning_rate=0.0002,
            gradient_clip_val=1.0,
            negative_slope=0.01,
            tta=tta,
            tb_logs=False,
            wandb_logs=False,
            wandb_project="brats",
            brats=True,
            deep_supervision=False,
            more_chn=False,
            invert_resampled_y=False,
            amp=True,
            benchmark=False,
            focal=False,
            save_ckpt=False,
            nfolds=5,
            seed=1,
            skip_first_n_eval=500,
            val_epochs=10,
            ckpt_path=None,
            ckpt_store_dir="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/checkpoints/",
            fold=0,
            patience=100,
            batch_size=4,
            val_batch_size=4,
            momentum=0.99,
            weight_decay=0.0001,
            save_preds=True,
            dim=3,
            resume_training=False,
            num_workers=8,
            epochs=2000,
            warmup=5,
            norm="instance",
            nvol=4,
            depth=5,
            min_fmap=4,
            deep_supr_num=2,
            res_block=False,
            filters=None,
            num_units=2,
            md_encoder=True,
            md_decoder=True,
            shape=False,
            paste=0,
            data2d_dim=3,
            oversampling=0.4,
            overlap=0.5,
            affinity="unique_contiguous",
            scheduler=False,
            optimizer="adam",
            blend="gaussian",
            train_batches=0,
            test_batches=0,
            swin=False,
            tpus=0,
        )

        self.model_paths = ["/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/checkpoints/0/last.ckpt"]
        self.args = []
        for i, pth in enumerate(self.model_paths):
            ckpt = torch.load(pth, map_location=self.device)
            ckpt["hyper_parameters"]["args"] = deepcopy(args)
            ckpt["hyper_parameters"]["args"].data="/opt/algorithm/data/11_3d/test"
            ckpt["hyper_parameters"]["args"].results="/opt/algorithm/results"
            ckpt["hyper_parameters"]["args"].config="/opt/algorithm/config/config.pkl"
            ckpt["hyper_parameters"][
                "args"
            ].ckpt_store_dir = "/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/checkpoints/" + str(i)
            ckpt["hyper_parameters"]["args"].ckpt_path = (
                "/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/checkpoints/" + str(i) + "/last.ckpt"
            )
            ckpt["hyper_parameters"]["args"].fold = i
            ckpt["hyper_parameters"]["args"].gpus = 1
            torch.save(ckpt, pth)
            self.args.append(ckpt["hyper_parameters"]["args"])

    def reslice(self, image, reference=None, target_spacing=[1.0, 1.0, 1.0]):
        if reference is not None:
            dims = image.GetDimension()
            resample = SimpleITK.ResampleImageFilter()
            resample.SetReferenceImage(reference)
            resample.SetInterpolator(SimpleITK.sitkLinear)
            resample.SetTransform(SimpleITK.AffineTransform(dims))

            resample.SetOutputSpacing(reference.GetSpacing())
            resample.SetSize(reference.GetSize())
            resample.SetOutputDirection(reference.GetDirection())
            resample.SetOutputOrigin(reference.GetOrigin())
            resample.SetDefaultPixelValue(0)

            newimage = resample.Execute(image)
            return newimage
        else:
            orig_spacing = image.GetSpacing()
            orig_size = image.GetSize()
            target_size = [
                int(round(osz * ospc / tspc))
                for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)
            ]
            return SimpleITK.Resample(
                image,
                target_size,
                SimpleITK.Transform(),
                SimpleITK.sitkLinear,
                image.GetOrigin(),
                target_spacing,
                image.GetDirection(),
                0,
                image.GetPixelID(),
            )

    def crf(self, image, pred):
        # image = np.transpose(image, [1,2,3,0])
        pair_energy = create_pairwise_bilateral(
            sdims=(1.0,) * 3, schan=(1.0,) * image.shape[-1], img=image, chdim=3
        )
        d = dcrf.DenseCRF(np.prod(image.shape[:-1]), pred.shape[0])
        U = unary_from_softmax(pred)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(pair_energy, compat=3)
        out = d.inference(5)
        out = np.asarray(out, np.float32).reshape(pred.shape)
        return out

    def nnunet_preprocess(self, image):
        os.makedirs(os.path.join("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde", "data/BraTS2021_train/imagesTs/"), exist_ok=True)
        SimpleITK.WriteImage(
            image,
            str(os.path.join("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde", "data/BraTS2021_train/imagesTs/BraTS2021_0001.nii.gz")),
        )
        data_desc = {
            "description": "Stroke Lesion Segmentation",
            "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
            "licence": "BLANK",
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"},
            "name": "BraTS2021_train",
            "numTest": 1,
            "numTraining": 0,
            "reference": "BLANK",
            "release": "BLANK",
            "tensorImageSize": "4D",
            "test": [
                os.path.join("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde", "data/BraTS2021_train/imagesTs/BraTS2021_0001.nii.gz")
            ],
            "training": [],
        }
        with open(os.path.join("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mde", "data/BraTS2021_train/dataset.json"), "w") as f:
            json.dump(data_desc, f)
        args = SimpleNamespace(
            data="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/data",
            results="/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/data",
            exec_mode="test",
            ohe=True,
            verbose=False,
            task="11",
            dim=3,
            n_jobs=1,
        )
        Preprocessor(args).run()

    def nnunet_infer(self, args):
        data_module = DataModule(args)
        data_module.setup()
        ckpt_path = verify_ckpt_path(args)
        model = NNUnet(args)
        callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]
        logger = False
        trainer = Trainer(
            logger=logger,
            default_root_dir=args.results,
            benchmark=True,
            deterministic=False,
            max_epochs=args.epochs,
            precision=16 if args.amp else 32,
            gradient_clip_val=args.gradient_clip_val,
            enable_checkpointing=args.save_ckpt,
            # callbacks=callbacks,
            callbacks=None,
            num_sanity_val_steps=0,
            accelerator="gpu",
            devices=args.gpus,
            num_nodes=args.nodes,
            strategy="ddp" if args.gpus > 1 else None,
            limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
            limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
            limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
            check_val_every_n_epoch=args.val_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        save_dir = os.path.join("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/prediction", str(args.fold))
        model.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        model.args = args
        trainer.test(
            model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=ckpt_path,
            verbose=False,
        )

    def nnunet_ensemble(self, paths, ref):
        preds = [np.load(f) for f in paths]
        pred = np.mean(preds, 0)
        pred = np.transpose(pred, (1,2,3,0))
        pred_image = SimpleITK.GetImageFromArray(pred, isVector=True)
        pred_image.SetOrigin(ref.GetOrigin())
        pred_image.SetSpacing(ref.GetSpacing())
        pred_image.SetDirection(ref.GetDirection())
        return pred_image

    def setup(self):
        os.makedirs("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/data", exist_ok=True)
        os.makedirs("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/results", exist_ok=True)
        os.makedirs("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/prediction", exist_ok=True)

    def cleanup(self):
        shutil.rmtree("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/data", ignore_errors=True)
        shutil.rmtree("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/results", ignore_errors=True)
        shutil.rmtree("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/prediction", ignore_errors=True)

    def process(self):
        inp_path = self._input_path  # Path for the input
        out_path = self._output_path # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            # dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            # im_shape = dat.shape
            # dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # # Convert 'dat' to Tensor, or as appropriate for your model.
            # ###########
            # ### Replace this section with the call to your code.
            # mean_dat = np.mean(dat)
            # dat[dat > mean_dat] = 1
            # dat[dat <= mean_dat] = 0
            # ###
            # ###########
            # dat = dat.reshape(*im_shape)
            # out_name = os.path.basename(fil)
            # out_filepath = os.path.join(out_path, out_name)
            # print(f"=== saving {out_filepath} from {fil} ===")
            # medpy.io.save(dat, out_filepath, hdr=hdr)

            t1w_image_1mm = SimpleITK.ReadImage(str(fil))
            # t1w_image_1mm = self.reslice(t1w_image)

            t1w_image_ref = t1w_image_1mm[...,0]

            # t1w_ss, t1w_mask = self.robex(t1w_image_1mm)
            # t1w_image_n4ss = self.n4(t1w_ss, t1w_mask)

            self.nnunet_preprocess(t1w_image_1mm)

            for args_ in self.args:
                self.nnunet_infer(args_)

            paths = [
                os.path.join(
                    "/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/prediction", str(i), "BraTS2021_0001.npy"
                )
                for i in range(len(self.args))
            ]
            prediction = self.nnunet_ensemble(paths, ref=t1w_image_ref)

            pred_crf = SimpleITK.GetArrayFromImage(prediction)
            # pred_crf = np.stack([1.0 - pred_crf, pred_crf])
            # img_crf = SimpleITK.GetArrayFromImage(t1w_image_1mm)
            # # img_crf = img_crf - img_crf.min(axis=(0, 1, 2))
            # # img_crf = 255 * (img_crf / img_crf.max(axis=(0, 1, 2)))
            # img_crf[img_crf < 0] = 0
            # img_crf[img_crf > 255] = 255
            # img_crf = np.asarray(img_crf, np.uint8)
            # pred_crf = np.asarray(pred_crf, np.float32)
            # # prediction = self.crf(img_crf, pred_crf)
            prediction = pred_crf
            pred = prediction

            pred = pred > 0.5

            # pred = remove_small_holes(pred, 50, 3)
            # pred = remove_small_objects(pred, 25, 3)

            self.cleanup()

            pred = np.argmax(pred, axis=-1)
            pred = pred.astype(int)

            prediction = SimpleITK.GetImageFromArray(pred)
            prediction.SetOrigin(t1w_image_ref.GetOrigin())
            prediction.SetSpacing(t1w_image_ref.GetSpacing())
            prediction.SetDirection(t1w_image_ref.GetDirection())

            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f"=== saving {out_filepath} from {fil} ===")

            SimpleITK.WriteImage(prediction, out_filepath)

        return


if __name__ == "__main__":
    pathlib.Path("/home/lchalcroft/mdunet/lka_paper_dockers/brats/mded/tumour-segmentation").mkdir(
        parents=True, exist_ok=True
    )
    ploras().process()
