ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

RUN apt-get update && apt-get -y install --reinstall build-essential && apt-get -y install git && apt-get -y install gcc

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN pip install --disable-pip-version-check --user -r requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110==1.16.0

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnunet /opt/algorithm/nnunet
COPY --chown=algorithm:algorithm utils /opt/algorithm/utils
COPY --chown=algorithm:algorithm checkpoints /opt/algorithm/checkpoints
COPY --chown=algorithm:algorithm config /opt/algorithm/config
COPY --chown=algorithm:algorithm data_loading /opt/algorithm/data_loading
COPY --chown=algorithm:algorithm data_preprocessing /opt/algorithm/data_preprocessing

# COPY --chown=algorithm:algorithm test /opt/algorithm/test
# COPY --chown=algorithm:algorithm test /input

ENTRYPOINT python -m process $0 $@
