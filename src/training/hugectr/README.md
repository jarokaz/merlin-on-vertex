# Using NVIDIA HugeCTR with Vertex Training

## Local testing


Cardinatilities based on day_0 - day_22

```
[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.task \
--num_epochs 0 \
--max_iter 5000 \
--eval_interval=1000 \
--display_interval=200 \
--snapshot_interval=4000 \
--batchsize=8192 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--workspace_size_per_gpu=300 \
--dropout_rate=0.5 \
--num_workers=12 \
--slot_size_array="[19615168, 35248, 17095, 7383, 20154, 4, 7075, 1404, 63, 13203982, 1077816, 300012, 11, 2209, 10942, 114, 4, 972, 15, 20417762, 5796980, 17498794, 379315, 12311, 103, 35]" \
--gpus="[[0,1,2,3]]" \
--model_dir=/criteo_data/model
```





## Submitting Vertex training jobs

### Build a training container

```
export PROJECT_ID=jk-mlops-dev
export IMAGE_NAME=gcr.io/${PROJECT_ID}/hugectr-training

docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME} 
```

### Create Vertex AI staging bucket 

```
export REGION=us-central1
export GCS_STAGING_BUCKET=gs://jk-vertex-merlin

gsutil mb -l ${REGION} ${GCS_STAGING_BUCKET}
```
### Submit a Vertex Training job

```
export VERTEX_SA="training-sa@jk-mlops-dev.iam.gserviceaccount.com"
export MACHINE_TYPE=a2-highgpu-4g
export ACCELERATOR_TYPE=NVIDIA_TESLA_A100
export ACCELERATOR_NUM=4

export TRAIN_DATA="/gcs/jk-criteo-bucket/criteo_processed_parquet/train/_file_list.txt"
export VALID_DATA="/gcs/jk-criteo-bucket/criteo_processed_parquet/valid/_file_list.txt"
export MAX_ITER=50000
export NUM_EPOCHS=0
export PER_GPU_BATCH_SIZE=2048
export SLOT_SIZE_ARRAY="[18792578,35176,17091,7383,20154,4,7075,1403,63,12687136,1054830,297377,11,2209,10933,113,4,972,15,19550853,5602712,16779972,375290,12292,101,35]"
export SNAPSHOT=0
export EVAL_INTERVAL=5000
export DISPLAY_INTERVAL=500
export WORKSPACE_SIZE_PER_GPU=61
export LR=0.001


python submit_vertex_job.py \
--project=$PROJECT \
--region=$REGION \
--gcs_bucket=$GCS_STAGING_BUCKET \
--vertex_sa=$VERTEX_SA \
--machine_type=$MACHINE_TYPE \
--accelerator_type=$ACCELERATOR_TYPE \
--accelerator_num=$ACCELERATOR_NUM \
--train_image=$IMAGE_NAME \
--train_data=$TRAIN_DATA \
--valid_data=$VALID_DATA \
--max_iter=$MAX_ITER \
--num_epochs=$NUM_EPOCHS \
--per_gpu_batchsize=$PER_GPU_BATCH_SIZE \
--snapshot=$SNAPSHOT \
--slot_size_array=$SLOT_SIZE_ARRAY \
--eval_interval=$EVAL_INTERVAL \
--display_interval=$DISPLAY_INTERVAL \
--workspace_size_per_gpu=$WORKSPACE_SIZE_PER_GPU \
--lr=$LR

```











