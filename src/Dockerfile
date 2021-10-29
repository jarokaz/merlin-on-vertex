# Latest version provided by NVIDIA (10/28/2021)
# Still Failing: NVTabular with T4 GPUs
FROM gcr.io/jk-mlops-dev/dongm-merlin-train-hugectr:latest

RUN pip install cloudml-hypertune

WORKDIR /src

COPY training/hugectr/* ./
