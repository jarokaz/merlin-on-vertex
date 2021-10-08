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

import os
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
import sys

IMAGE_URI = os.environ['IMAGE_URI']

@dsl.component(
    base_image=IMAGE_URI
)
def convert_csv_to_parquet_op(
    output_datasets: Output[Dataset],
    train_paths: list,
    valid_paths: list,
    output_converted: str,
    columns: list,
    cols_dtype: dict,
    sep: str,
    gpus: str,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False
):
    '''
    Component to convert CSV file(s) to Parquet format using NVTabular.

    output_datasets: Output[Dataset]
        Output metadata with references to the converted CSVs in GCS.
        Usage:
            output_datasets.metadata['train']
                .example: 'gs://my_bucket/folders/train'
            output_datasets.metadata['valid']
                .example: 'gs://my_bucket/folders/valid'
    train_paths: list
        List of paths to folders or files in GCS for training.
        For recursive folder search, set the recursive variable to True
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
            'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    valid_paths: list
        List of paths to folders or files in GCS for validation.
        For recursive folder search, set the recursive variable to True
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
            'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    output_converted: str
        Path in GCS to write the converted parquet files.
        Format:
            '/gcs/<bucket_name>/<subfolder1>/<subfolder>'
    columns: list
        List with the columns name from CSV file.
        Format:
            ['I1', 'I2', ..., 'C1', ...]
    cols_dtype: dict
        Dict with the dtype of the columns from CSV.
        Format:
            {'I1':'int32', ..., 'C20':'hex'}
    gpus: str
        GPUs available. 
        Format:
            If there are 4 gpus available, must be '0,1,2,3'
    shuffle: str
        How to shuffle the converted CSV, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''

    # Standard Libraries
    import logging
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

    logging.info('Creating a Dask CUDA cluster')
    cluster = LocalCUDACluster(
        rmm_pool_size=get_rmm_size(0.8 * device_mem_size())
    )
    client = Client(cluster)

    for folder_name, data_paths in zip(
        ['train', 'valid'], 
        [train_paths, valid_paths]
    ):
        valid_paths = []
        for path in data_paths:
            try:
                if fs_spec.isfile(path):
                    valid_paths.append(path)
                else:
                    path = os.path.join(path, rec_symbol)
                    for i in fs_spec.glob(path):
                        if fs_spec.isfile(i):
                            valid_paths.append(f'gs://{i}')
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
            client=client,
            assume_missing=True
        )

        full_output_path = os.path.join('/gcs', output_converted, folder_name)

        logging.info(f'Writing parquet file(s) to {full_output_path}')
        if shuffle:
            shuffle = getattr(Shuffle, shuffle)

        dataset.to_parquet(
            full_output_path,
            preserve_files=True,
            shuffle=shuffle
        )

        # Write output path to metadata
        output_datasets.metadata[folder_name] = os.path.join(
            'gs://', output_converted, folder_name
        )


@dsl.component(
    base_image=IMAGE_URI
)
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
    Component to generate statistics from the dataset.

    datasets: Input[Dataset]
        Input metadata with references to the train and valid converted
        datasets in GCS.
        Usage:
            full_path_train = datasets.metadata.get('train')
                .example: 'gs://my_bucket/folders/converted/train'
            full_path_valid = datasets.metadata.get('valid')
                .example: 'gs://my_bucket/folders/converted/valid'
    fitted_workflow: Output[Artifact]
        Output metadata with the path to the fitted workflow artifacts
        (statistics) and converted datasets in GCS.
        Usage:
            fitted_workflow.metadata['fitted_workflow']
                .example: '/gcs/my_bucket/fitted_workflow'
            fitted_workflow.metadata['datasets']
                .example: 'gs://my_bucket/folders/converted/train'
    workflow_path: str
        Path to the current workflow, not fitted. This path must have 
        2 files: metadata.json and workflow.pkl.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>'
    split_name: str
        Which dataset split to calculate the statistics. 'train' or 'valid'
    '''

    import logging
    import nvtabular as nvt
    import os
    
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from nvtabular.utils import device_mem_size

    logging.basicConfig(level=logging.INFO)

    # Check if the `split_name` dataset is present
    logging.info(f'Checking if split {split_name} is present.')
    data_path = datasets.metadata.get(split_name, '')
    if not data_path:
        raise RuntimeError(f'Dataset does not have {split_name} split.')

    # Dask Cluster defintions
    device_size = device_mem_size()
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)
    rmm_pool_size = (device_pool_size // 256) * 256

    logging.info('Creating a Dask CUDA cluster')
    cluster = LocalCUDACluster(
        device_memory_limit=device_limit,
        rmm_pool_size=rmm_pool_size
    )
    client = Client(cluster)

    # Load Transformation steps
    FIT_FOLDER = os.path.join('/gcs', workflow_path, 'fitted_workflow')

    logging.info('Loading saved workflow')
    workflow = nvt.Workflow.load(
        os.path.join('/gcs', workflow_path), client
    )
    fitted_dataset = nvt.Dataset(
        os.path.join(data_path, '*.parquet'),
        engine="parquet", 
        part_size=part_size
    )
    logging.info('Starting workflow fitting')
    workflow.fit(fitted_dataset)
    logging.info('Finished generating statistics for dataset.')

    logging.info(f'Saving workflow to {FIT_FOLDER}')
    workflow.save(FIT_FOLDER)

    fitted_workflow.metadata['fitted_workflow'] = FIT_FOLDER
    fitted_workflow.metadata['datasets'] = datasets.metadata


@dsl.component(
    base_image=IMAGE_URI
)
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
    Component to transform a dataset according to the workflow specifications.

    fitted_workflow: Input[Artifact]
        Input metadata with the path to the fitted_workflow and the 
        location of the converted datasets in GCS (train and validation).
        Usage:
            fitted_workflow.metadata['datasets']['train']
                example: 'gs://my_bucket/converted/train'
            fitted_workflow.metadata['fitted_workflow']
                example: '/gcs/my_bucket/fitted_workflow'
    transformed_dataset: Output[Dataset]
        Output metadata with the path to the transformed dataset 
        and the validation dataset.
        Usage:
            transformed_dataset.metadata['transformed_dataset']
                .example: 'gs://my_bucket/transformed_data/train'
            transformed_dataset.metadata['original_datasets']
                .example: 'gs://my_bucket/converted/train'
    output_transformed: str,
        Path in GCS to write the transformed parquet files.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
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

    logging.info('Creating a Dask CUDA cluster')
    cluster = LocalCUDACluster(
        device_memory_limit=device_limit,
        rmm_pool_size=rmm_pool_size
    )
    client = Client(cluster)

    # Load Transformation steps
    logging.info('Loading workflow and statistics')
    workflow = nvt.Workflow.load(
        fitted_workflow.metadata.get('fitted_workflow'), client
    )

    logging.info('Creating dataset definition')
    dataset = nvt.Dataset(
        os.path.join(data_path, '*.parquet'), 
        engine="parquet", 
        part_size=part_size
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

    transformed_dataset.metadata['transformed_dataset'] = \
        os.path.join('gs://', output_transformed, split_name)
    transformed_dataset.metadata['original_datasets'] = \
        fitted_workflow.metadata.get('datasets')


@dsl.component(
    base_image=IMAGE_URI
)
def export_parquet_from_bq_op(
    output_datasets: Output[Dataset],
    output_converted: str,
    bq_project: str,
    bq_dataset_id: str,
    bq_table_train: str,
    bq_table_valid: str,
    location: str
):
    '''
    Component to export PARQUET files from a bigquery table.

    output_datasets: dict
        Output metadata with the GCS path for the exported datasets.
        Usage:
            output_datasets.metadata['train']
                .example: 'gs://bucket_name/subfolder/train/'
    output_converted: str
        Path to write the exported parquet files.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
    bq_project: str
        GCP project id
        Format:
            'my_project'
    bq_dataset_id: str
        Bigquery dataset id
        Format:
            'my_dataset_id'
    bq_table_train: str
        Bigquery table name for training dataset
        Format:
            'my_train_table_id'
    bq_table_valid: str
        BigQuery table name for validation dataset
        Format:
            'my_valid_table_id'
    '''

    import logging
    import os
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    extract_job_config = bigquery.ExtractJobConfig()
    extract_job_config.destination_format = 'PARQUET'

    client = bigquery.Client(project=bq_project)
    dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_id)

    for folder_name, table_id in zip(
        ['train', 'valid'], 
        [bq_table_train, bq_table_valid]
    ):
        bq_glob_path = os.path.join(
            'gs://', 
            output_converted,
            folder_name,
            f'{folder_name}-*.parquet'
        )
        table_ref = dataset_ref.table(table_id)

        logging.info(f'Extracting {table_ref} to {bq_glob_path}')
        extract_job = client.extract_table(
            table_ref, 
            bq_glob_path, 
            location=location,
            job_config=extract_job_config
        )
        extract_job.result()
        
        full_output_path = os.path.join('gs://', output_converted, folder_name)
        logging.info(
            f'Saving metadata for {folder_name} path: {full_output_path}'
        )
        output_datasets.metadata[folder_name] = full_output_path
    
    logging.info('Finished exporting to GCS.')
