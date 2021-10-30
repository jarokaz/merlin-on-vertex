```
docker run -it --rm --gpus all --cap-add SYS_NICE \
--network host \
-e AIP_STORAGE_URI='gs://jk-merlin-dev/ensembles/triton-ensemble-20211029205018' \
gcr.io/jk-mlops-dev/triton-hugectr-inference:latest
```

```
curl -X POST \
   http://localhost:8000/v2/models/deepfm_ens/infer \
  -d @criteo.json
```  
  
  
```
docker run -it --rm --gpus all --cap-add SYS_NICE \
--network host \
-v /home/jupyter/staging:/staging \
gcr.io/merlin-on-gcp/dongm-merlin-inference-hugectr:latest
```
  
```
MODEL_REPOSITORY=gs://jk-merlin-dev/ensembles/ensemble_1
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 tritonserver --model-repository=$MODEL_REPOSITORY \
--backend-config=hugectr,ps=gs://jk-merlin-dev/ensembles/ensemble_1/ps.json
```


