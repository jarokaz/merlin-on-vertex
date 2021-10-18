# Using NVIDIA HugeCTR with Vertex Training

## Local testing


Cardinatilities based on day_0 - day_22 in `criteo_parquet_processed`

```
[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]
```

Cardinalities based on day_0 - day_22 in criteo_parquet_processed_0.6_t1

```
[19615168, 35248, 17095, 7383, 20154, 4, 7075, 1404, 63, 13203982, 1077816, 300012, 11, 2209, 10942, 114, 4, 972, 15, 20417762, 5796980, 17498794, 379315, 12311, 103, 35]
```




```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
-e AIP_MODEL_DIR='/criteo_data/model_t1' \
-e AIP_CHECKPOINT_DIR='/criteo_data/checkpoints' \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.task \
--num_epochs 0 \
--max_iter 200000 \
--max_eval_batches 200 \
--eval_batches 1000 \
--eval_interval=500 \
--display_interval=200 \
--snapshot_interval=0 \
--batchsize=8192 \
--train_data=/criteo_data/criteo_processed_parquet_0.6_t1/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet_0.6_t1/valid/_file_list.txt  \
--workspace_size_per_gpu=300 \
--dropout_rate=0.5 \
--num_workers=12 \
--slot_size_array="[19615168, 35248, 17095, 7383, 20154, 4, 7075, 1404, 63, 13203982, 1077816, 300012, 11, 2209, 10942, 114, 4, 972, 15, 20417762, 5796980, 17498794, 379315, 12311, 103, 35]" \
--gpus="[[0,1,2,3]]" 
```


```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/training:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
-e AIP_MODEL_DIR='/criteo_data/model' \
-e AIP_CHECKPOINT_DIR='/criteo_data/checkpoints' \
nvcr.io/nvidia/merlin/merlin-inference:21.09 \
python -m trainer.task \
--num_epochs 0 \
--max_iter 5000 \
--max_eval_batches 200 \
--eval_batches 1000 \
--eval_interval=500 \
--display_interval=100 \
--snapshot_interval=0 \
--batchsize=8192 \
--train_data=/criteo_data/criteo_processed_parquet_0.6/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet_0.6/valid/_file_list.txt  \
--workspace_size_per_gpu=300 \
--dropout_rate=0.5 \
--num_workers=12 \
--slot_size_array="[19615168, 35248, 17095, 7383, 20154, 4, 7075, 1404, 63, 13203982, 1077816, 300012, 11, 2209, 10942, 114, 4, 972, 15, 20417762, 5796980, 17498794, 379315, 12311, 103, 35]" \
--gpus="[[0,1,2,3]]" 
```

















