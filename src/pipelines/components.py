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
"""KFP components."""

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
from . import config


@dsl.component(
    base_image=config.IMAGE_URI
)
def convert_csv_to_parquet_op(
    output_dataset: Output[Dataset],
    data_paths: list,
    split: str,
    output_dir: str,
    sep: str,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
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
    output_dir: str
        Path in GCS to write the converted parquet files.
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>'
    recursive: bool
        If it must recursivelly look for files in path.
    shuffle: str
        How to shuffle the converted CSV, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''
    # Standard Libraries
    import logging
    import os

    # ETL library
    from preprocessing import etl

    logging.basicConfig(level=logging.INFO)

    logging.info('Getting column names and dtypes')
    col_dtypes = etl.get_criteo_col_dtypes()

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers = n_workers,
        device_limit_frac = device_limit_frac,
        device_pool_frac = device_pool_frac
    )

    logging.info(f'Creating {split} dataset.')
    dataset = etl.create_csv_dataset(
        data_paths=data_paths,
        sep=sep,
        recursive=recursive, 
        col_dtypes=col_dtypes,
        part_mem_frac=part_mem_frac, 
        client=client
    )

    fuse_output_dir = os.path.join(
        output_dir.replace('gs://', '/gcs/'), split
    )
    
    logging.info(f'Writing parquet file(s) to {fuse_output_dir}')
    etl.convert_csv_to_parquet(fuse_output_dir, dataset, shuffle)

    # Write output path to metadata
    output_dataset.metadata['dataset_path'] = os.path.join(output_dir, split)
    output_dataset.metadata['split'] = split


@dsl.component(
    base_image=config.IMAGE_URI
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    workflow_path: str,
    n_workers: int,
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
    workflow: Output[Artifact]
        Output metadata with the path to the fitted workflow artifacts
        (statistics) and converted datasets in GCS.
        Usage:
            workflow.metadata['workflow']
                .example: '/gcs/my_bucket/fitted_workflow'
            workflow.metadata['datasets']
                .example: 'gs://my_bucket/folders/converted/train'
    workflow_path: str
        Path to write the fitted workflow.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>'
    split_name: str
        Which dataset split to calculate the statistics. 'train' or 'valid'
    '''
    from preprocessing import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers = n_workers,
        device_limit_frac = device_limit_frac, 
        device_pool_frac = device_pool_frac
    )

    # Create data transformation workflow. This step will only 
    # calculate statistics based on the transformations
    logging.info('Creating transformation workflow.')
    criteo_workflow = etl.create_criteo_nvt_workflow(client=client)

    # Create dataset to be fitted
    logging.info(f'Creating dataset to be analysed.')
    dataset = etl.create_parquet_dataset(
        client=client,
        data_path=parquet_dataset.metadata['dataset_path'],
        part_mem_frac=part_mem_frac
    )

    split = parquet_dataset.metadata['split']

    logging.info(f'Starting workflow fitting for {split} split.')
    criteo_workflow = etl.analyze_dataset(criteo_workflow, dataset)
    logging.info('Finished generating statistics for dataset.')

    workflow_path_fuse = workflow_path.replace('gs://', '/gcs/')
    etl.save_workflow(criteo_workflow, workflow_path_fuse)
    logging.info('Workflow saved to GCS')

    workflow.metadata['workflow'] = workflow_path_fuse


@dsl.component(
    base_image=config.IMAGE_URI
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    transformed_output_dir: str,
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.8,
    device_pool_frac: float = 0.9,
    part_mem_frac: float = 0.125,
):
    '''
    Component to transform a dataset according to the workflow specifications.

    workflow: Input[Artifact]
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
    transformed_output_dir: str,
        Path in GCS to write the transformed parquet files.
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/'
    '''
    from preprocessing import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_cluster(
        n_workers=n_workers,
        device_limit_frac=device_limit_frac, 
        device_pool_frac=device_pool_frac
    )

    logging.info('Loading workflow and statistics')
    criteo_workflow = etl.load_workflow(
        workflow_path=workflow.metadata['workflow'],
        client=client
    )

    split = parquet_dataset.metadata['split']

    logging.info(f'Creating dataset definition for {split} split')
    dataset = etl.create_parquet_dataset(
        client=client,
        data_path=parquet_dataset.metadata['dataset_path'],
        part_mem_frac=part_mem_frac
    )

    logging.info('Workflow is loaded')
    logging.info('Starting workflow transformation')
    dataset = etl.transform_dataset(
        dataset=dataset,
        workflow=criteo_workflow
    )

    # Define output path for transformed files
    transformed_fuse_dir = os.path.join(
        transformed_output_dir.replace('gs://', '/gcs/'), 
        split
    )

    logging.info('Applying transformation')
    etl.save_dataset(dataset, transformed_fuse_dir)

    transformed_dataset.metadata['dataset_path'] = os.path.join(
        transformed_output_dir, split
    )
    transformed_dataset.metadata['split'] = split


@dsl.component(
    base_image=config.IMAGE_URI
)
def export_parquet_from_bq_op(
    output_dataset: Output[Dataset],
    bq_project: str,
    bq_location: str,
    bq_dataset_name: str,
    bq_table_name: str,
    split: str,
    output_dir: str
):
    '''
    Component to export PARQUET files from a bigquery table.

    output_datasets: dict
        Output metadata with the GCS path for the exported datasets.
        Usage:
            output_datasets.metadata['train']
                .example: 'gs://bucket_name/subfolder/train/'
    output_dir: str
        Path to write the exported parquet files.
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/'
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
    from preprocessing import etl
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    client = bigquery.Client(project=bq_project)
    dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_name)

    full_output_path = os.path.join(output_dir, split)
    
    logging.info(
        f'Extracting {bq_table_name} table to {full_output_path} path.'
    )
    etl.extract_table_from_bq(
        client=client,
        output_dir=full_output_path,
        dataset_ref=dataset_ref,
        table_id=bq_table_name,
        location=bq_location
    )

    # Write output path to metadata
    output_dataset.metadata['dataset_path'] = os.path.join(output_dir, split)
    output_dataset.metadata['split'] = split

    logging.info('Finished exporting to GCS.')
