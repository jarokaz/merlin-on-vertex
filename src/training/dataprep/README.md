## Data preparation 

This is a temporary set of scripts to locally preprocess the Criteo dataset for HugeCTR (and other) training. The samples will be obsoleted when the NVT KFP pipelines are ready.

All the processing is done locally on data in PD.

### Convert the original Criteo dataset in the TSV format to parquet

```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-on-vertex/src/training/dataprep:/src \
-w /src \
-v /home/jupyter/data:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python convert_to_parquet.py \
--input_path /criteo_data/criteo_tsv \
--output_path /criteo_data/criteo_parquet \
--dask_path /criteo_data/dask-workspace \
--devices 0,1,2,3 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--part_mem_frac 0.125
```

### Preprocess the Criteo parquet files
```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-on-vertex/src/training/dataprep:/src \
-w /src \
-v /home/jupyter/data:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python preprocess.py \
--train_folder /criteo_data/criteo_raw_parquet_train \
--valid_folder /criteo_data/criteo_raw_parquet_valid \
--output_folder /criteo_data/criteo_processed_parquet_0.6_t1 \
--devices 0,1,2,3 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 4 \
--part_mem_frac 0.08 \
--out_files_per_proc 8 \
--freq_limit 6 \
--shuffle PER_PARTITION
```

Cardinatilities based on day_0 - day_22

```
[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]
```

### Reshard the preprocessed files

```

```


```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-on-vertex/src/training/dataprep:/src \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python convert_to_parquet.py \
--input_path /criteo_data/criteo_tsv \
--output_path /criteo_data/criteo_raw_parquet \
--dask_path /criteo_data/dask-workspace \
--devices 0,1,2,3 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--part_mem_frac 0.125
```

```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-on-vertex/src/training/dataprep:/src \
-w /src \
-v /home/jupyter/data:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python preprocess-fsspec.py \
--train_folder gs://jk-criteo-bucket/criteo_raw_parquet_train \
--valid_folder gs://jk-criteo-bucket/criteo_raw_parquet_valid \
--output_folder gs://jk-criteo-bucket/tttttttt/criteo_processed_parquet \
--devices 0,1 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 4 \
--part_mem_frac 0.08 \
--out_files_per_proc 8 \
--freq_limit 6 \
--shuffle PER_PARTITION
```



```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-on-vertex/src/training/dataprep:/src \
-w /src \
-v /home/jupyter/data:/criteo_data \
nvcr.io/nvidia/merlin/merlin-inference:0.6 \
python preprocess.py \
--train_folder /criteo_data/criteo_raw_parquet_train \
--valid_folder /criteo_data/criteo_raw_parquet_valid \
--output_folder /criteo_data/criteo_processed_parquet_0.6_t1 \
--devices 0,1,2,3 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 4 \
--part_mem_frac 0.08 \
--out_files_per_proc 8 \
--freq_limit 6 \
--shuffle PER_PARTITION
```