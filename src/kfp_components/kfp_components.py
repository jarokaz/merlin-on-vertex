# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing components"""

from kfp.v2 import dsl
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input, 
    InputPath, 
    Model, 
    Output,
    OutputPath
)

from typing import Optional

BASE_IMAGE_NAME = 'us-east1-docker.pkg.dev/renatoleite-mldemos/docker-images/nvt-conda'

@dsl.component(base_image=BASE_IMAGE_NAME)
def convert_csv_to_parquet_op(
    output_datasets: Output[Dataset],
    train_paths: list,
    valid_paths: list,
    output_path: str,
    columns: list,
    cols_dtype: list,
    sep: str,
    gpus: str,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False
):
    '''
    train_paths: list
        List of paths to folders or files in GCS for training.
        For recursive folder search, set the recursive variable to True
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/' or
            '<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    valid_paths: list
        List of paths to folders or files in GCS for validation
        For recursive folder search, set the recursive variable to True
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/' or
            '<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    output_path: str
        Path to write the converted parquet files
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
    gpus: str
        GPUs available. Example:
            If there are 4 gpus available, must be '0,1,2,3'
    output_datasets: dict
        Metadata pointing to the converted dataset
        Format:
            output_datasets.metadata['train'] = \
                '<bucket_name>/<subfolder1>/<subfolder>/'
    shuffle: str
        How to shuffle the converted data, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''

    # Standard Libraries
    import logging
    from pathlib import Path
    import fsspec
    import os

    # External Dependencies
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import numpy as np

    # NVTabular
    from nvtabular.utils import device_mem_size, get_rmm_size
    import nvtabular as nvt
    from nvtabular.io.shuffle import Shuffle

    logging.basicConfig(level=logging.INFO)

    # Specify column dtypes (from numpy). Note that 'hex' means that
    # the values will be hexadecimal strings that should be converted to int32
    logging.info('Converting columns dtypes to numpy objects')
    converted_col_dtype = {}
    for col, dt in cols_dtype.items():
        if dt == 'hex':
            converted_col_dtype[col] = 'hex'
        else:
            converted_col_dtype[col] = getattr(np, dt)

    fs_spec = fsspec.filesystem('gs')
    rec_symbol = '**' if recursive else '*'
    TRAIN_SPLIT_FOLDER = 'train'
    VALID_SPLIT_FOLDER = 'valid'

    if gpus:
        logging.info('Creating a Dask CUDA cluster')
        cluster = LocalCUDACluster(
                    n_workers=len(gpus.split(sep=',')),
                    CUDA_VISIBLE_DEVICES=gpus,
                    rmm_pool_size=get_rmm_size(0.8 * device_mem_size())
        )
        client = Client(cluster)
    else:
        raise Exception('Cannot create Cluster. \
                    Provide a list of available GPUs')

    for folder_name, data_paths in zip(
        [TRAIN_SPLIT_FOLDER, VALID_SPLIT_FOLDER], 
        [train_paths, valid_paths]
    ):
        valid_paths = []
        for path in data_paths:
            try:
                if fs_spec.isfile(path):
                    valid_paths.append(os.path.join('/gcs', path))
                else:
                    path = os.path.join(path, rec_symbol)
                    for i in fs_spec.glob(path):
                        if fs_spec.isfile(i):
                            valid_paths.append(os.path.join('/gcs', i))
            except FileNotFoundError as fnf_expt:
                print(fnf_expt)
                print('One of the paths provided are incorrect.')
            except OSError as os_err:
                print(os_err)
                print(f'Verify access to the bucket.')

        dataset = nvt.Dataset(
            path_or_source = valid_paths,
            engine='csv',
            names=columns,
            sep=sep,
            dtypes=converted_col_dtype,
            client=client
        )

        full_output_path = os.path.join('/gcs', output_path, folder_name)

        logging.info(f'Writing parquet file(s) to {full_output_path}')
        if shuffle:
            shuffle = getattr(Shuffle, shuffle)

        dataset.to_parquet(
            full_output_path,
            preserve_files=True,
            shuffle=shuffle
        )
        output_datasets.metadata[folder_name] = full_output_path
        
    # Maybe this can speedup steps transition
    # TODO: Check if this makes any difference.
    client.close()


@dsl.component(base_image=BASE_IMAGE_NAME)
def fit_dataset_op(
    datasets: Input[Dataset],
    fitted_workflow: Output[Artifact],
    workflow_path: str,
    gpus: str,
    split_name: Optional[str] = 'train',
    protocol: Optional[str] = 'tcp',
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
    '''
    datasets: dict
        Input metadata from previus step. Stores the full path of the 
        converted datasets.
        How to access:
            full_path = datasets.metadata.get('train')
    fitted_workflow: dict
        Output metadata for next step. Stores the full path of the 
        converted dataset, and saved workflow with statistics.
    workflow_path: str
        Path to the current workflow, not fitted.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
    split_name: str
        Which dataset to calculate the statistics. 'train' or 'valid'
    '''

    import logging
    import nvtabular as nvt
    import os
    
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from nvtabular.utils import device_mem_size

    logging.basicConfig(level=logging.INFO)

    FIT_FOLDER = os.path.join('/gcs', workflow_path, 'fitted_workflow')

    # Check if the `split_name` dataset is present
    data_path = datasets.metadata.get(split_name, '')
    if not data_path:
        raise RuntimeError(f'Dataset does not have {split_name} split.')

    # Dask Cluster defintions
    device_size = device_mem_size()
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)
    rmm_pool_size = (device_pool_size // 256) * 256

    if gpus:
        logging.info('Creating a Dask CUDA cluster')
        cluster = LocalCUDACluster(
            protocol=protocol,
            n_workers=len(gpus.split(sep=',')),
            CUDA_VISIBLE_DEVICES=gpus,
            device_memory_limit=device_limit,
            rmm_pool_size=rmm_pool_size
        )
        client = Client(cluster)
    else:
        raise Exception('Cannot create Cluster. \
                            Provide a list of available GPUs')

    # Load Transformation steps
    full_workflow_path = os.path.join('/gcs', workflow_path)

    logging.info('Loading saved workflow')
    workflow = nvt.Workflow.load(full_workflow_path, client)
    fitted_dataset = nvt.Dataset(
        data_path, engine="parquet", part_size=part_size
    )
    logging.info('Starting workflow fitting')
    workflow.fit(fitted_dataset)
    logging.info('Finished generating statistics for dataset.')

    logging.info(f'Saving workflow to {FIT_FOLDER}')
    workflow.save(FIT_FOLDER)

    fitted_workflow.metadata['fitted_workflow'] = FIT_FOLDER
    fitted_workflow.metadata['datasets'] = datasets
    
    # Maybe this can speedup steps transition
    # TODO: Check if this makes any difference.
    client.close()


@dsl.component(base_image=BASE_IMAGE_NAME)
def transform_dataset_op(
    fitted_workflow: Input[Artifact],
    transformed_dataset: Output[Dataset],
    output_transformed: str,
    gpus: str,
    split_name: str = 'train',
    shuffle: str = None,
    protocol: str = 'tcp',
    device_limit_frac: float = 0.8,
    device_pool_frac: float = 0.9,
    part_mem_frac: float = 0.125,
):
    '''
    fitted_workflow: dict
        Input metadata from previous step. Stores the path of the fitted_workflow
        and the location of the datasets (train and validation).
        Usage:
            train_path = fitted_workflow.metadata['datasets']['train]
            output: '<bucket_name>/<subfolder1>/<subfolder>/'
    transformed_dataset: dict
        Output metadata for next step. Stores the path of the transformed dataset 
        and the validation dataset.
    output_transformed: str,
        Path to write the transformed parquet files
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
    gpus: str
        GPUs available. Example:
            If there are 4 gpus available, must be '0,1,2,3'
    shuffle: str
        How to shuffle the converted data, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''
    import logging
    import nvtabular as nvt
    import os
    
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from nvtabular.utils import device_mem_size
    from nvtabular.io.shuffle import Shuffle

    logging.basicConfig(level=logging.INFO)

    # Define output path for transformed files
    TRANSFORM_FOLDER = os.path.join('/gcs', output_transformed, split_name)

    # Get path to dataset to be transformed
    data_path = fitted_workflow.metadata.get('datasets').get(split_name, '')
    if not data_path:
        raise RuntimeError(f'Dataset does not have {split_name} split.')

    # Dask Cluster defintions
    device_size = device_mem_size()
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)
    rmm_pool_size = (device_pool_size // 256) * 256

    if gpus:
        logging.info('Creating a Dask CUDA cluster')
        cluster = LocalCUDACluster(
            protocol=protocol,
            n_workers=len(gpus.split(sep=',')),
            CUDA_VISIBLE_DEVICES=gpus,
            device_memory_limit=device_limit,
            rmm_pool_size=rmm_pool_size
        )
        client = Client(cluster)
    else:
        raise Exception('Cannot create Cluster. \
                            Provide a list of available GPUs')

    # Load Transformation steps
    logging.info('Loading workflow and statistics')
    workflow = nvt.Workflow.load(
        fitted_workflow.metadata.get('fitted_workflow'), client
    )

    logging.info('Creating dataset definition')
    dataset = nvt.Dataset(
        data_path, engine="parquet", part_size=part_size
    )

    if shuffle:
        shuffle = getattr(Shuffle, shuffle)

    logging.info('Starting workflow transformation')
    workflow.transform(dataset).to_parquet(
        output_files=len(gpus.split(sep='/')),
        output_path=TRANSFORM_FOLDER,
        shuffle=shuffle
    )
    logging.info('Finished transformation')

    transformed_dataset.metadata['transformed_dataset'] = TRANSFORM_FOLDER
    transformed_dataset.metadata['original_datasets'] = \
        fitted_workflow.metadata.get('datasets')

    # Maybe this can speedup steps transition
    # TODO: Check if this makes any difference.
    client.close()


@dsl.component(base_image=BASE_IMAGE_NAME)
def export_parquet_from_bq_op(
    output_datasets: Output[Dataset],
    output_path: str,
    bq_project: str,
    bq_dataset_id: str,
    bq_table_train: str,
    bq_table_valid: str,
    location: str
):
    '''
    output_path: str
        Path to write the exported parquet files
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/'
    bq_project: str
        GCP project id
    bq_dataset_id: str
        Bigquery dataset id
    bq_table_train: str
        Bigquery table name for training dataset
    bq_table_valid: str
        BigQuery table name for validation dataset
    output_datasets: dict
        Output metadata for the next step. Stores the path in GCS
        for the datasets.
        Usage:
            train_path = output_datasets.metadata['train']
            # returns: bucket_name/subfolder/subfolder/
    '''

    import logging
    import os
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    TRAIN_SPLIT_FOLDER = 'train'
    VALID_SPLIT_FOLDER = 'valid'

    client = bigquery.Client()
    dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_id)

    for folder_name, table_id in zip(
        [TRAIN_SPLIT_FOLDER, VALID_SPLIT_FOLDER], 
        [bq_table_train, bq_table_valid]
    ):
        bq_glob_path = os.path.join(
            'gs://',
            output_path,
            folder_name,
            f'{folder_name}-*.parquet'
        )
        table_ref = dataset_ref.table(table_id)

        logging.info(f'Extracting {table_ref} to {bq_glob_path}')
        client.extract_table(table_ref, bq_glob_path, location=location)

        full_output_path = os.path.join('/gcs', output_path, folder_name)
        logging.info(
            f'Saving metadata for {folder_name} path: {full_output_path}'
        )
        output_datasets.metadata[folder_name] = full_output_path
    
    logging.info('Finished exporting to GCS.')