## Train model

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
-e AIP_MODEL_DIR='/criteo_data/model' \
-e AIP_CHECKPOINT_DIR='/criteo_data/checkpoints' \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.task \
--num_epochs 0 \
--max_iter 250000 \
--max_eval_batches 200 \
--eval_batches 2000 \
--eval_interval=1000 \
--display_interval=200 \
--snapshot_interval=0 \
--batchsize=8192 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--workspace_size_per_gpu=300 \
--dropout_rate=0.5 \
--num_workers=12 \
--slot_size_array="[19615168, 35248, 17095, 7383, 20154, 4, 7075, 1404, 63, 13203982, 1077816, 300012, 11, 2209, 10942, 114, 4, 972, 15, 20417762, 5796980, 17498794, 379315, 12311, 103, 35]" \
--gpus="[[0,1,2,3]]" 
```


## Inference environment

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
--network host \
-v /home/jupyter/merlin-on-vertex/src/:/src \
-v /home/jupyter/data:/criteo_data \
-v /home/jupyter/data/models:/models \
-w /src \
-e AIP_STORAGE_URI='gs://jk-criteo-bucket/model_registry' \
gcr.io/jk-mlops-dev/merlin-inference:latest
```


```
docker build -t gcr.io/jk-mlops-dev/merlin-inference:latest -f Dockerfile.triton .
docker push gcr.io/jk-mlops-dev/merlin-inference:latest
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
--network host \
-e AIP_STORAGE_URI='gs://jk-criteo-bucket/models' \
gcr.io/jk-mlops-dev/merlin-inference:latest
```



Start triton
```
tritonserver --model-repository=/models \
--backend-config=hugectr,ps=/models/ps.json 
```

