docker run -it --rm --gpus all --cap-add SYS_NICE \
--network host \
-e AIP_STORAGE_URI='gs://jk-merlin-dev/ensembles/triton-ensemble-20211029205018' \
gcr.io/jk-mlops-dev/triton-hugectr-inference:latest



curl -X POST \
   http://localhost:8000/v2/models/deepfm_ens/infer \
  -d @criteo.json
  