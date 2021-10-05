# HUGECTR debugging

To replicate a crash during training on a `a2-highgpu-8g`.

## Provision a Vertex AI notebook instance 

Use the following settings:

- Use `a2-highgpu-8g` machine type
- Use TensorFlow Enterprise 2.6 image
- Set boot disk to 500GB
- Set data disk to 3000GB
- Install GPU driver automatically

# Clone this repo

Log on to JupyterLab and open a Jupyter terminal. 

```
cd
git clone https://github.com/jarokaz/merlin-on-vertex.git

```



## Get the Criteo dataset

Download the preprocessed Criteo dataset.

```
cd
mkdir data
cd data
gsutil -m cp -r gs://workshop-datasets/criteo_processed_parquet .
```

## Run tests

First run the test using 4 GPUs. This should work.

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/vertex_training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.train \
--num_epochs 0 \
--max_iter 50000 \
--eval_interval=5000 \
--batchsize=8192 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--display_interval=500 \
--workspace_size_per_gpu=61 \
--slot_size_array="[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]" \
--gpus="[[0,1,2,3]]"
```

Then run the test using 8 GPUs. This crashes.

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/vertex_training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.train \
--num_epochs 0 \
--max_iter 50000 \
--eval_interval=5000 \
--batchsize=16384 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--display_interval=500 \
--workspace_size_per_gpu=61 \
--slot_size_array="[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]" \
--gpus="[[0,1,2,3,4,5,6,7]]"
```
