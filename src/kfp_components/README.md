# Run docker container

docker run -it --rm --gpus all \
-v /home/renatoleite/workspace/merlin-on-vertex/src:/src \
-v /home/renatoleite/data:/gcs \
-v /home/renatoleite/output:/output \
-w /src \
7461 /bin/bash

# Mount FUSE

gcsfuse --implicit-dirs renatoleite-criteo-partial /gcs


BASE_IMAGE = 'nvcr.io/nvidia/merlin/merlin-training'