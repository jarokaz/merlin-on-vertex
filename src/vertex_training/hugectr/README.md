# Using NVIDIA HugeCTR with Vertex Training


## Build a training container

```
export PROJECT_ID=jk-mlops-dev
export IMAGE_NAME=gcr.io/${PROJECT_ID}/hugectr-training

docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME} 
```

## Create Vertex AI staging bucket 

```
export REGION=us-central1
export GCS_STAGING_BUCKET=gs://jk-vertex-merlin

gsutil mb -l ${REGION} ${GCS_STAGING_BUCKET}
```
## Submit a Vertex Training job

```
export VERTEX_SA="training-sa@jk-mlops-dev.iam.gserviceaccount.com"
export MACHINE_TYPE=a2-highgpu-2g
export ACCELERATOR_TYPE=NVIDIA_TESLA_A100
export ACCELERATOR_NUM=2

export TRAIN_DATA="/gcs/jk-criteo-bucket/criteo_processed/output/train/_file_list.txt"
export VALID_DATA="/gcs/jk-criteo-bucket/criteo_processed/output/valid/_file_list.txt"
export MAX_ITER=5000
export NUM_EPOCHS=0
export PER_GPU_BATCH_SIZE=2048
export SLOT_SIZE_ARRAY="[2839307,28141,15313,7229,19673,4,6558,1297,63,2156343,327548,178478,11,2208,9517,73,4,957,15,2893928,1166099,2636476,211349,10776,92,35]"
export SNAPSHOT=0
export EVAL_INTERVAL=1000
export DISPLAY_INTERVAL=200
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



## Local testing


Categorical features cardinalities for day_0 - day_2 datasets


```
[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]

'2839307,28141,15313,7229,19673,4,6558,1297,63,2156343,327548,178478,11,2208,9517,73,4,957,15,2893928,1166099,2636476,211349,10776,92,35'
```



docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/vertex_training/hugectr:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.train \
--num_epochs 1 \
--max_iter 50000 \
--eval_interval=600 \
--batchsize=8192 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=200 \
--workspace_size_per_gpu=1 \
--slot_size_array="[2839307,28141,15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]" \
--gpus="[[0,1]]"
```



```
docker build -t gcr.io/jk-mlops-dev/merlin-train .
docker push gcr.io/jk-mlops-dev/merlin-train
```


```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src/merlin-sandbox:/src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/hugectr/train/criteo_parquet.py
```

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/merlin-sandbox/hugectr/train/train.py
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=100000 \
--eval_interval=1000 \
--batchsize=2048 \
--train_data=/criteo_data/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_data/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=2000 \
--display_interval=500 \
--gpus=0,1
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=5000 \
--batchsize=4096 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_data/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=9000 \
--display_interval=1000 \
--gpus=0,1
```

```
```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/criteo_processed:/criteo_processed \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=5000 \
--batchsize=2048 \
--snapshot=0 \
--train_data=/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=9000 \
--display_interval=1000 \
--gpus=0
```
```



```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=5000 \
--eval_interval=500 \
--batchsize=2048 \
--train_data=gs://jk-vertex-us-central1/criteo_data/train/_file_list.txt \
--valid_data=gs://jk-vertex-us-central1/criteo_data/valid/_file_list.txt \
--gpus=0,1
```



## Create filestore

```
gcloud beta filestore instances create nfs-server \
--zone=us-central1-a \
--tier=BASIC_SDD \
--file-share=name="vol1",capacity=2TB \
--network=name="default"
```


# Train


```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/dcn_parquet.py 
```



```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=7000 \
--batchsize=4096 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=1000 \
--workspace_size_per_gpu=2000 \
--gpus=0,1
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python -m trainer.task \
--num_epochs 1 \
--max_iter 50000 \
--eval_interval=600 \
--batchsize=4096 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=200 \
--workspace_size_per_gpu=1 \
--slot_size_array="[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]" \
--gpus="[[0]]"
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE --network host \
-v /home/jupyter/criteo_processed:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python -m trainer.task \
--num_epochs 1 \
--max_iter 50000 \
--eval_interval=600 \
--batchsize=16384 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=200 \
--workspace_size_per_gpu=1 \
--slot_size_array="[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]" \
--gpus="[[0,1,2,3]]"
```