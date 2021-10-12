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

IMAGE_URI = os.environ['IMAGE_URI']

@dsl.component(
    base_image=IMAGE_URI
)
def convert_csv_to_parquet_op(
    output_datasets: Output[Dataset],
    train_paths: list,
    valid_paths: list,
    output_converted: str,
    sep: str,
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
    from etl import etl

    logging.basicConfig(level=logging.INFO)

    logging.info('Getting column names and dtypes')
    col_dtypes = etl.get_criteo_col_dtypes()

    logging.info('Creating a Dask CUDA cluster')
    client = etl.create_convert_cluster()

    for folder_name, data_paths in zip(
        ['train', 'valid'], 
        [train_paths, valid_paths]
    ):
        logging.info(f'Creating {folder_name} dataset.')
        dataset = etl.create_csv_dateset(
            data_paths=data_paths,
            sep=sep,
            recursive=recursive, 
            col_dtypes=col_dtypes, 
            client=client
        )

        full_output_path = os.path.join('/gcs', output_converted, folder_name)
        logging.info(f'Writing parquet file(s) to {full_output_path}')
        etl.convert_csv_to_parquet(full_output_path, dataset, shuffle)

        # Write output path to metadata
        output_datasets.metadata[folder_name] = os.path.join(
            'gs://', output_converted, folder_name
        )


@dsl.component(
    base_image=IMAGE_URI
)
def fit_dataset_op(
    datasets: Input[Dataset],
    workflow: Output[Artifact],
    workflow_path: str,
    split_name: Optional[str] = 'train',
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
        Path to the fitted workflow.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>'
    split_name: str
        Which dataset split to calculate the statistics. 'train' or 'valid'
    '''
    from etl import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    # Retrieve `split_name` from metadata
    data_path = datasets.metadata[split_name]

    # Create data transformation workflow. This step will only 
    # calculate statistics based on the transformations
    criteo_workflow = etl.create_criteo_nvt_workflow()    

    # Create Dask cluster
    client = etl.create_transform_cluster(device_limit_frac, device_pool_frac)

    # Create dataset to be fitted
    dataset = etl.create_fit_dataset(
        data_path=data_path,
        part_mem_frac=part_mem_frac,
        client=client
    )

    full_workflow_path = os.path.join('/gcs', workflow_path)

    logging.info('Starting workflow fitting')
    etl.fit_and_save_workflow(criteo_workflow, dataset, full_workflow_path)
    logging.info('Finished generating statistics for dataset.')
    logging.info(f'Workflow saved to {full_workflow_path}')

    workflow.metadata['workflow'] = full_workflow_path
    workflow.metadata['datasets'] = datasets.metadata


@dsl.component(
    base_image=IMAGE_URI
)
def transform_dataset_op(
    workflow: Input[Artifact],
    transformed_dataset: Output[Dataset],
    output_transformed: str,
    split_name: str = 'train',
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
    output_transformed: str,
        Path in GCS to write the transformed parquet files.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/'
    '''
    from etl import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    # Define output path for transformed files
    transform_folder = os.path.join('/gcs', output_transformed, split_name)

    # Get path to dataset to be transformed
    data_path = workflow.metadata['datasets'][split_name]

    # Create Dask cluster
    client = etl.create_transform_cluster(device_limit_frac, device_pool_frac)

    logging.info('Creating dataset definition')
    logging.info('Loading workflow and statistics')
    logging.info('Starting workflow transformation')
    etl.workflow_transform(
        data_path=data_path,
        part_mem_frac=part_mem_frac,
        client=client,
        workflow_path=workflow.metadata['workflow'],
        destination_transformed=transform_folder,
        shuffle=shuffle
    )
    logging.info('Finished transformation')

    transformed_dataset.metadata['transformed_dataset'] = \
        os.path.join('gs://', output_transformed, split_name)
    transformed_dataset.metadata['original_datasets'] = \
        workflow.metadata.get('datasets')


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
    from etl import etl
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    client = bigquery.Client(project=bq_project)
    dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_id)

    for folder_name, table_id in zip(
        ['train', 'valid'], 
        [bq_table_train, bq_table_valid]
    ):
        logging.info(f'Extracting {table_id}')
        etl.extract_table_from_bq(
            client=client,
            output_converted=output_converted,
            folder_name=folder_name,
            dataset_ref=dataset_ref,
            table_id=table_id,
            location=location
        )
        
        full_output_path = os.path.join('gs://', output_converted, folder_name)
        logging.info(
            f'Saving metadata for {folder_name} path: {full_output_path}'
        )
        output_datasets.metadata[folder_name] = full_output_path
    
    logging.info('Finished exporting to GCS.')