from weakref import ref
import SimpleITK
import numpy as np
import json
import os
from pathlib import Path

DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")

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

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# todo change with your team-name
class ploras:
    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH,
    ):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path(
                "/home/lchalcroft/mdunet/miccai22_dockers/docker-isles/test/"
            )
            self._output_path = Path(
                "/home/lchalcroft/mdunet/miccai22_dockers/docker-isles/output/"
            )
            self._algorithm_output_path = (
                self._output_path / "stroke-lesion-segmentation"
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
            data="/opt/algorithm/data/15_3d/test",
            results="/opt/algorithm/results",
            config="/opt/algorithm/config/config.pkl",
            logname="ploras",
            task="15",
            gpus=1,
            nodes=1,
            learning_rate=0.0002,
            gradient_clip_val=1.0,
            negative_slope=0.01,
            tta=tta,
            tb_logs=False,
            wandb_logs=False,
            wandb_project="isles",
            brats=False,
            deep_supervision=True,
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
            ckpt_store_dir="/opt/algorithm/checkpoints/",
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
            md_encoder=False,
            md_decoder=False,
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
            swin=True,
            tpus=0,
        )

        self.model_paths = ["/opt/algorithm/checkpoints/0/last.ckpt"]
        self.args = []
        for i, pth in enumerate(self.model_paths):
            ckpt = torch.load(pth, map_location=self.device)
            ckpt["hyper_parameters"]["args"] = deepcopy(args)
            for key in args.keys():
                ckpt["hyper_parameters"]["args"][key] = args[key]
            ckpt["hyper_parameters"][
                "args"
            ].ckpt_store_dir = "/opt/algorithm/checkpoints/" + str(i)
            ckpt["hyper_parameters"]["args"].ckpt_path = (
                "/opt/algorithm/checkpoints/" + str(i) + "/last.ckpt"
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
        os.makedirs("/opt/algorithm/data/ISLES2022/imagesTs/", exist_ok=True)
        SimpleITK.WriteImage(
            image, str("/opt/algorithm/data/ISLES2022/imagesTs/ISLES2022_0001.nii.gz")
        )
        data_desc = {
            "description": "Stroke Lesion Segmentation",
            "labels": {"0": "Background", "1": "Lesion"},
            "licence": "BLANK",
            "modality": {"0": "ADC", "1": "DWI", "2": "FLAIR"},
            "name": "ISLES2022",
            "numTest": 1,
            "numTraining": 0,
            "reference": "BLANK",
            "release": "BLANK",
            "tensorImageSize": "4D",
            "test": [
                "/opt/algorithm/data/ISLES2022/imagesTs/ISLES2022_0001.nii.gz",
            ],
            "training": [],
        }
        with open("/opt/algorithm/data/ISLES2022/dataset.json", "w") as f:
            json.dump(data_desc, f)
        args = SimpleNamespace(
            data="/opt/algorithm/data",
            results="/opt/algorithm/data",
            exec_mode="test",
            ohe=False,
            verbose=False,
            task="15",
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
        save_dir = os.path.join("/opt/algorithm/prediction", str(args.fold))
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
        pred = np.mean(preds, 0)[1]
        pred_image = SimpleITK.GetImageFromArray(pred)
        pred_image.SetOrigin(ref.GetOrigin())
        pred_image.SetSpacing(ref.GetSpacing())
        pred_image.SetDirection(ref.GetDirection())
        return pred_image

    def setup(self):
        os.makedirs("/opt/algorithm/data", exist_ok=True)
        os.makedirs("/opt/algorithm/results", exist_ok=True)
        os.makedirs("/opt/algorithm/prediction", exist_ok=True)

    def cleanup(self):
        shutil.rmtree("/opt/algorithm/data")
        shutil.rmtree("/opt/algorithm/results")
        shutil.rmtree("/opt/algorithm/prediction")

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image',
                        'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = (
            input_data["dwi_image"],
            input_data["adc_image"],
            input_data["flair_image"],
        )

        # Get all json inputs.
        dwi_json, adc_json, flair_json = (
            input_data["dwi_json"],
            input_data["adc_json"],
            input_data["flair_json"],
        )

        self.setup()

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        dwi_image_rs = self.reslice(dwi_image, reference=flair_image)
        adc_image_rs = self.reslice(adc_image, reference=flair_image)
        flair_image_rs = self.reslice(flair_image, reference=flair_image)
        dwi_image_1mm = self.reslice(dwi_image_rs)
        adc_image_1mm = self.reslice(adc_image_rs)
        flair_image_1mm = self.reslice(flair_image_rs)

        dwi_image_data = SimpleITK.GetArrayFromImage(dwi_image_1mm)
        adc_image_data = SimpleITK.GetArrayFromImage(adc_image_1mm)
        flair_image_data = SimpleITK.GetArrayFromImage(flair_image_1mm)

        img = np.stack([adc_image_data, dwi_image_data, flair_image_data], axis=-1)
        stack_image = SimpleITK.GetImageFromArray(img, isVector=True)
        stack_image.SetOrigin(flair_image_1mm.GetOrigin()), stack_image.SetSpacing(
            flair_image_1mm.GetSpacing()
        ), stack_image.SetDirection(flair_image_1mm.GetDirection())

        self.nnunet_preprocess(stack_image)

        for args_ in self.args:
            self.nnunet_infer(args_)

        paths = [
            os.path.join("/opt/algorithm/prediction", str(i), "ISLES2022_0001.npy")
            for i in range(len(self.args))
        ]
        prediction = self.nnunet_ensemble(paths, ref=flair_image_1mm)

        pred_crf = SimpleITK.GetArrayFromImage(prediction)
        pred_crf = np.stack([1.0 - pred_crf, pred_crf])
        img_crf = SimpleITK.GetArrayFromImage(stack_image)
        img_crf = img_crf - img_crf.min(axis=(0, 1, 2))
        img_crf = 255 * (img_crf / img_crf.max(axis=(0, 1, 2)))
        img_crf[img_crf < 0] = 0
        img_crf[img_crf > 255] = 255
        img_crf = np.asarray(img_crf, np.uint8)
        pred_crf = np.asarray(pred_crf, np.float32)
        #        prediction = self.crf(img_crf, pred_crf)
        prediction = pred_crf
        prediction = prediction[1]
        prediction = SimpleITK.GetImageFromArray(prediction)

        prediction.SetOrigin(dwi_image_1mm.GetOrigin()), prediction.SetSpacing(
            dwi_image_1mm.GetSpacing()
        ), prediction.SetDirection(dwi_image_1mm.GetDirection())

        prediction = self.reslice(prediction, reference=dwi_image_rs)
        prediction = self.reslice(prediction, reference=dwi_image)

        prediction = SimpleITK.GetArrayFromImage(prediction)

        prediction[prediction > 1] = 0

        prediction = prediction > 0.5

        #        prediction = remove_small_holes(prediction, 10, 1)
        #        prediction = remove_small_objects(prediction, 2, 1)

        self.cleanup()

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = (
            input_data["dwi_image"].GetOrigin(),
            input_data["dwi_image"].GetSpacing(),
            input_data["dwi_image"].GetDirection(),
        )

        # Segment images.
        prediction = self.predict(input_data)  # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(
            spacing
        ), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {
                "outputs": [
                    dict(
                        type="Image",
                        slug="stroke-lesion-segmentation",
                        filename=str(output_image_path.name),
                    )
                ],
                "inputs": [
                    dict(type="Image", slug="dwi-brain-mri", filename=input_filename)
                ],
            }

            self._case_results.append(json_result)
            self.save()

    def load_isles_case(self):
        """Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty."""

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug="dwi-brain-mri", filetype="image")
        adc_image_path = self.get_file_path(slug="adc-brain-mri", filetype="image")
        flair_image_path = self.get_file_path(slug="flair-brain-mri", filetype="image")

        # Get MR metadata paths.
        dwi_json_path = self.get_file_path(
            slug="dwi-mri-acquisition-parameters", filetype="json"
        )
        adc_json_path = self.get_file_path(slug="adc-mri-parameters", filetype="json")
        flair_json_path = self.get_file_path(
            slug="flair-mri-acquisition-parameters", filetype="json"
        )

        input_data = {
            "dwi_image": SimpleITK.ReadImage(str(dwi_image_path)),
            "dwi_json": json.load(open(dwi_json_path)),
            "adc_image": SimpleITK.ReadImage(str(adc_image_path)),
            "adc_json": json.load(open(adc_json_path)),
            "flair_image": SimpleITK.ReadImage(str(flair_image_path)),
            "flair_json": json.load(open(flair_json_path)),
        }

        # Set input information.
        input_filename = str(dwi_image_path).split("/")[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype="image"):
        """Gets the path for each MR image/json file."""

        if filetype == "image":
            file_list = list((self._input_path / "images" / slug).glob("*.mha"))
        elif filetype == "json":
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print("Loading error")
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    ploras().process()
