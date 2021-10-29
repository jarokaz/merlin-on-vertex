# Latest version provided by NVIDIA (10/28/2021)
# Still Failing: NVTabular with T4 GPUs
FROM nvcr.io/nvidia/merlin/merlin-training:21.09

RUN pip install cloudml-hypertune

WORKDIR /src

COPY training/hugectr/* ./
