# Testing GCS performance with NVTabular

## Provision a test VM

From Cloud Shell

```
export PROJECT_ID=jk-mlops-dev
export INSTANCE_NAME="merlin-dev"
export VM_IMAGE_PROJECT="deeplearning-platform-release"
export VM_IMAGE_FAMILY="common-cu110"
export MACHINE_TYPE="a2-highgpu-2g"
export BUCKET_REGION="us-central1"
export BUCKET_NAME=gs://jk-criteo-bucket
export LOCATION="us-central1-c"
export ACCELERATOR_TYPE=NVIDIA_TESLA_A100
export ACCELERATOR_COUNT=2
export BOOT_DISK_SIZE=500



gcloud notebooks instances create $INSTANCE_NAME \
--location=$LOCATION \
--vm-image-project=$VM_IMAGE_PROJECT \
--vm-image-family=$VM_IMAGE_FAMILY \
--machine-type=$MACHINE_TYPE \
--accelerator-type=$ACCELERATOR_TYPE \
--accelerator-core-count=$ACCELERATOR_COUNT \
--boot-disk-size=$BOOT_DISK_SIZE \
--install-gpu-driver

```

After your instance has been created connect to JupyterLab and open a JupyterLab terminal.


## Prepare Criteo data

### Create a GCS bucket in the same region as your notebook instance

```
gsutil mb -l $BUCKET_REGION $BUCKET_NAME
```

### Copy Criteo parquet files
```
gsutil -m cp -r gs://workshop-datasets/criteo-parque $BUCKET_NAME/

```



## Run a benchmark

### Clone the repo
```
cd 
git clone https://github.com/merlin-on-vertex
cd merlin-on-vertex/nvtabular_benchmark

```

### Build a container

```
docker build -t nvt-test .
```

```
docker run -it --rm --gpus all \
-v /tmp:/out \
nvt-test \
python dask-nvtabular-criteo-benchmark.py \
--data-path gs://jk-criteo-bucket/criteo-parque \
--out-path gs://jk-criteo-bucket/test_output \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile /out/dask-report.html
```