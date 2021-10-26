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
    base_image=config.NVT_IMAGE_URI
)
def convert_csv_to_parquet_op(
    output_dataset: Output[Dataset],
    data_paths: list,
    split: str,
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

    output_dataset: Output[Dataset]
        Output metadata with references to the converted CSVs in GCS.
        Usage:
            output_dataset.path
    data_paths: list
        List of paths to folders or files in GCS for training.
        For recursive folder search, set the recursive variable to True
        Format:
            'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
            'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    recursive: bool
        If it must recursivelly look for files in path.
    shuffle: str
        How to shuffle the converted CSV, default to None.
        Options:
            PER_PARTITION
            PER_WORKER
            FULL
    '''
    
    import logging
    import os
    from preprocessing import etl
    import feature_utils

    logging.basicConfig(level=logging.INFO)

    logging.info('Getting column names and dtypes')
    col_dtypes = feature_utils.get_criteo_col_dtypes()

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
        output_dataset.path.replace('gs://', '/gcs/'), split
    )
    
    logging.info(f'Writing parquet file(s) to {fuse_output_dir}')
    etl.convert_csv_to_parquet(fuse_output_dir, dataset, shuffle)

    # Write metadata
    output_dataset.metadata['split'] = split


@dsl.component(
    base_image=config.NVT_IMAGE_URI
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    n_workers: int,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
    '''
    Component to generate statistics from the dataset.

    parquet_dataset: Input[Dataset]
        Input metadata with references to the train and valid converted
        datasets in GCS.
    workflow: Output[Artifact]
        Output metadata with the path to the fitted workflow artifacts
        (statistics) and converted datasets in GCS.
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
        data_path=parquet_dataset.path,
        part_mem_frac=part_mem_frac
    )

    split = parquet_dataset.metadata['split']

    logging.info(f'Starting workflow fitting for {split} split.')
    criteo_workflow = etl.analyze_dataset(criteo_workflow, dataset)
    logging.info('Finished generating statistics for dataset.')

    workflow_path_fuse = workflow.path.replace('gs://', '/gcs/')
    etl.save_workflow(criteo_workflow, workflow_path_fuse)
    logging.info('Workflow saved to GCS')


@dsl.component(
    base_image=config.NVT_IMAGE_URI
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
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
    transformed_dataset: Output[Dataset]
        Output metadata with the path to the transformed dataset 
        and the validation dataset.
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
    
    workflow_fuse_path = workflow.path.replace("gs://", "/gcs/")

    logging.info('Loading workflow and statistics')
    criteo_workflow = etl.load_workflow(
        workflow_path=workflow_fuse_path,
        client=client
    )

    split = parquet_dataset.metadata['split']

    logging.info(f'Creating dataset definition for {split} split')
    dataset = etl.create_parquet_dataset(
        client=client,
        data_path=parquet_dataset.path,
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
        transformed_dataset.path.replace('gs://', '/gcs/'), 
        split
    )

    logging.info('Applying transformation')
    etl.save_dataset(dataset, transformed_fuse_dir)

    transformed_dataset.metadata['split'] = split


@dsl.component(
    base_image=config.NVT_IMAGE_URI
)
def export_parquet_from_bq_op(
    output_dataset: Output[Dataset],
    bq_project: str,
    bq_location: str,
    bq_dataset_name: str,
    bq_table_name: str,
    split: str,
):
    '''
    Component to export PARQUET files from a bigquery table.

    output_datasets: dict
        Output metadata with the GCS path for the exported datasets.
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

    full_output_path = os.path.join(output_dataset.path, split)
    
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

    # Write metadata
    output_dataset.metadata['split'] = split

    logging.info('Finished exporting to GCS.')
    
    
@dsl.component
def train_hugectr_op(
    transformed_train_dataset: Input[Dataset],
    transformed_valid_dataset: Input[Dataset],
    model: Output[Model],
    project: str,
    region: str,
    service_account: str,
    job_display_name: str,
    training_image_url: str,
    replica_count: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    num_workers: int,
    per_gpu_batch_size: int,
    max_iter: int,
    max_eval_batches: int,
    eval_batches: int,
    dropout_rate: int,
    lr: int ,
    num_epochs: int,
    eval_interva: int,
    snapshot: int,
    display_interval: int
):
    
    import logging
    from google.cloud import aiplatform as vertex_ai

    vertex_ai.init(
        project=project,
        location=region
    )
    
    train_data_fuse = os.path.join(
        transformed_train_dataset.path, '_file_list.txt').replace("gs://", "/gcs/")
    valid_data_fuse = os.path.join(
        transformed_valid_dataset.path, '_file_list.txt').replace("gs://", "/gcs/")
    schema_path = os.path.join(
        transformed_train_dataset.path, "schema.pbtxt").replace("gs://", "/gcs/")
    
    gpus = json.dumps([list(range(accelerator_count))]).replace(' ','')
                 
    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": training_image_url,
                "command": ["python", "-m", "trainer.task"],
                "args": [
                    f'--per_gpu_batch_size={per_gpu_batch_size}',
                    f'--train_data={train_data_fuse}', 
                    f'--valid_data={valid_data_fuse}',
                    f'--schema={schema_path}',
                    f'--max_iter={max_iter}',
                    f'--max_eval_batches={max_eval_batches}',
                    f'--eval_batches={eval_batches}',
                    f'--dropout_rate={dropout_rate}',
                    f'--lr={lr}',
                    f'--num_workers={num_workers}',
                    f'--num_epochs={num_epochs}',
                    f'--eval_interval={eval_interval}',
                    f'--snapshot={snapshot}',
                    f'--display_interval={display_interval}',
                    f'--gpus={gpus}',
                ],
            },
        }
    ]
    
    logging.info('worker_pool_specs:')
    logging.info(worker_pool_specs)
    
    logging.info("Submitting a custom job to Vertex AI...")
    job = vertex_ai.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=model.path
    )
    
    job.run(
        sync=True,
        service_account=service_account,
        restart_job_on_worker_restart=False
    )
    
    logging.info("Custom Vertex AI job completed.")


@dsl.component(
    base_image=config.NVT_IMAGE_URI
)  
def export_triton_ensemble(
    model: Input[Model],
    workflow: Input[Artifact],
    exported_model: Output[Model]
):
  
    import logging
    from serving import export
    import feature_utils
    
    model_location_fuse = model.path.replace("gs://", "/gcs/")
    workflow_location_fuse =  workflow.path.replace("gs://", "/gcs/")
    output_location_fuse = exported_model.path.replace("gs://", "/gcs/")
    
    logging.info('Exporting Triton ensemble model...')
    export.export_ensemble(
        workflow_path=workflow_location_fuse,
        saved_model_path=model_location_fuse,
        output_path=output_location_fuse,
        categotical_columns=feature_utils.categotical_columns(),
        continuous_columns=feature_utils.continuous_columns(),
        label_columns=feature_utils.label_columns()
    )
    logging.info('Triton model exported.')
    
    
def upload_vertex_model(
    exported_model: Input[Artifact],
    uploaded_model: Output[Artifact],
    project: str,
    region: str,
    display_name: str,
    serving_container_image_uri: str,
    serving_container_environment_variables: dict,
    labels: dict
):
    
    import logging
    from google.cloud import aiplatform as vertex_ai
    
    vertex_ai.init(project=project, location=region)
    
    exported_model_path = exported_model.path
    
    logging.info(f"Exported model location: {exported_model_path}")

    vertex_model = vertex_ai.Model.upload(
        display_name=display_name,
        artifact_uri=exported_model_path,
        serving_container_image_uri=serving_container_image_uri,
        labels=labels
    )

    model_uri = vertex_model.gca_resource.name
    logging.info(f"Model uploaded to Vertex AI: {model_uri}")
    uploaded_model.set_string_custom_property("model_uri", model_uri)
    
    
    
    
    
    
    
