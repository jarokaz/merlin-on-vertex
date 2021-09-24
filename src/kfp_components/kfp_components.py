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
    Component to convert CSV file(s) to Parquet format using NVTabular.

    output_datasets: Output[Dataset]
        Output metadata with references to the converted CSVs in GCS.
        Usage:
            output_datasets.metadata['train']
                .example: '/gcs/my_bucket/folders/train'
            output_datasets.metadata['valid']
                .example: '/gcs/my_bucket/folders/valid'
    train_paths: list
        List of paths to folders or files in GCS for training.
        For recursive folder search, set the recursive variable to True
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/' or
            '<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    valid_paths: list
        List of paths to folders or files in GCS for validation.
        For recursive folder search, set the recursive variable to True
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>/' or
            '<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
            a combination of both.
    output_path: str
        Path in GCS to write the converted parquet files.
        Format:
            '<bucket_name>/<subfolder1>/<subfolder>'
    columns: list
        List with the columns name from CSV file.
        Format:
            ['I1', 'I2', ..., 'C1', ...]
    cols_dtype: list
        List with the dtype of the columns from CSV. The position of the 
        dtype in the list must match the position of the column name
        in the columns variable explained before.
        Format:
            ['int32', ..., 'float']
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
        raise Exception(
            'Cannot create Cluster. Provide a list of available GPUs'
        )

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

        # Write output path to metadata
        output_datasets.metadata[folder_name] = full_output_path


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
    Component to generate statistics from the dataset.

    datasets: Input[Dataset]
        Input metadata with references to the train and valid converted
        datasets in GCS.
        Usage:
            full_path_train = datasets.metadata.get('train')
                .example: '/gcs/my_bucket/folders/converted/train'
            full_path_valid = datasets.metadata.get('train')
                .example: '/gcs/my_bucket/folders/converted/valid'
    fitted_workflow: Output[Artifact]
        Output metadata with the path to the fitted workflow artifacts
        (statistics) and converted datasets in GCS.
        Usage:
            fitted_workflow.metadata['fitted_workflow']
                .example: '/gcs/my_bucket/fitted_workflow'
            fitted_workflow.metadata['datasets']
                .example: '/gcs/my_bucket/folders/converted/train'
    workflow_path: str
        Path to the current workflow, not fitted. This folder must have 
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

    FIT_FOLDER = os.path.join('/gcs', workflow_path, 'fitted_workflow')

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
        raise Exception(
            'Cannot create Cluster. Provide a list of available GPUs'
        )

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
    fitted_workflow.metadata['datasets'] = datasets.metadata


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
    Component to transform a dataset according to the workflow specifications.

    fitted_workflow: Input[Artifact]
        Input metadata with the path to the fitted_workflow and the 
        location of the converted datasets in GCS (train and validation).
        Usage:
            fitted_workflow.metadata['datasets']['train']
                example: '/gcs/my_bucket/converted/train'
            fitted_workflow.metadata['fitted_workflow']
                example: '/gcs/my_bucket/fitted_workflow'
    transformed_dataset: Output[Dataset]
        Output metadata with the path to the transformed dataset 
        and the validation dataset.
        Usage:
            transformed_dataset.metadata['transformed_dataset']
                .example: '/gcs/my_bucket/transformed_data/train'
            transformed_dataset.metadata['original_datasets']
                .example: '/gcs/my_bucket/converted/train'
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
        raise Exception(
            'Cannot create Cluster. Provide a list of available GPUs'
        )

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
    Component to export PARQUET files from a bigquery table.

    output_datasets: dict
        Output metadata with the GCS path for the exported datasets.
        Usage:
            output_datasets.metadata['train']
                .example: '/gcs/bucket_name/subfolder/train/'
    output_path: str
        Path to write the exported parquet files. Note it must 
        start with gs://.
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
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    TRAIN_SPLIT_FOLDER = 'train'
    VALID_SPLIT_FOLDER = 'valid'

    extract_job_config = bigquery.ExtractJobConfig()
    extract_job_config.destination_format = 'PARQUET'

    client = bigquery.Client(project=bq_project)
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
        extract_job = client.extract_table(
            table_ref, 
            bq_glob_path, 
            location=location,
            job_config=extract_job_config
        )
        extract_job.result()
        
        full_output_path = os.path.join('/gcs', output_path, folder_name)
        logging.info(
            f'Saving metadata for {folder_name} path: {full_output_path}'
        )
        output_datasets.metadata[folder_name] = full_output_path
    
    logging.info('Finished exporting to GCS.')


@dsl.component(base_image=BASE_IMAGE_NAME)
def import_parquet_to_bq_op(
    transformed_dataset: Input[Dataset],
    output_bq_table: Output[Dataset],
    bq_project: str,
    bq_dataset_id: str,
    bq_dest_table_id: str
):
    '''
    Component to load PARQUET files to a Bigquery table.

    transformed_dataset: dict
        Input metadata. Stores the path in GCS
        for the datasets.
        Usage:
            train_path = output_dataset['train']
            # returns: bucket_name/subfolder/subfolder/
    bq_project: str
        GCP project id
        Format:
            'my_project'
    bq_dataset_id: str
        Bigquery dataset id
        Format:
            'my_dataset_id'
    bq_dest_table_id: str
        Bigquery destination table name
        Format:
            'my_destination_table_id'
    '''

    # Standard Libraries
    import logging
    import os
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)

    data_path = transformed_dataset.metadata['transformed_dataset'][5:]
    full_data_path = os.path.join('gs://', data_path, '*.parquet')
    
    # Construct a BigQuery client object.
    client = bigquery.Client(project=bq_project)
    table_id = '.'.join([bq_project, bq_dataset_id, bq_dest_table_id])

    job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)

    load_job = client.load_table_from_uri(
        full_data_path, table_id, job_config=job_config
    )  # Make an API request.

    logging.info('Loading data from GCS to BQ')
    load_job.result()  # Waits for the job to complete.

    output_bq_table.metadata['bq_project'] = bq_project
    output_bq_table.metadata['bq_dataset_id'] = bq_dataset_id
    output_bq_table.metadata['bq_dest_table_id'] = bq_dest_table_id
    output_bq_table.metadata['dataset_path'] = data_path


@dsl.component(base_image=BASE_IMAGE_NAME)
def load_bq_to_feature_store_op(
    output_bq_table: Input[Dataset],
    feature_store_path: Output[Artifact],
    columns: list,
    cols_dtype: list
):
    '''
    Component to create a feature store and load the data from Bigquery.

    output_bq_table: Input[Artifact]
        Input metadata with references to the project ID, dataset ID and table
        ID where BQ table was imported.
        Usage:
            output_bq_table.metadata['bq_project']
                .example: 'my_project'
            output_bq_table.metadata['bq_dataset_id']
                .example: 'my_dataset'
            output_bq_table.metadata['bq_dest_table_id']
                .example: 'my_table_id'
            output_bq_table.metadata['dataset_path']
                .example: 'my_bucket/subfolder/train'
    feature_store_path: Output[Dataset]
        Output metadata with informations about the feature store,
        its entity types and features.
        Usage:
            feature_store_path.metadata['featurestore_id']
                .example: 'feat_store_1234'
    columns: list
        List with the columns name from CSV file.
        Format:
            ['I1', 'I2', ..., 'C1', ...]
    cols_dtype: list
        List with the dtype of the columns from CSV. The position of the 
        dtype in the list must match the position of the column name
        in the columns variable explained before.
        Format:
            ['int32', ..., 'float']
    '''

    from datetime import datetime
    import re
    import time
    import logging
    import os

    from google.api_core.exceptions import AlreadyExists

    from google.cloud.aiplatform_v1beta1 import (
        FeaturestoreOnlineServingServiceClient, FeaturestoreServiceClient)
    from google.cloud.aiplatform_v1beta1.types import \
        entity_type as entity_type_pb2
    from google.cloud.aiplatform_v1beta1.types import feature as feature_pb2
    from google.cloud.aiplatform_v1beta1.types import \
        featurestore as featurestore_pb2
    from google.cloud.aiplatform_v1beta1.types import \
        featurestore_service as featurestore_service_pb2
    from google.cloud.aiplatform_v1beta1.types import io as io_pb2
    from google.protobuf.timestamp_pb2 import Timestamp

    logging.basicConfig(level=logging.INFO)

    PROJECT_ID = output_bq_table.metadata['bq_project']
    DATASET_ID = output_bq_table.metadata['bq_dataset_id']
    TABLE_ID = output_bq_table.metadata['bq_dest_table_id']

    # To import the BQ data to feature store we need to 
    # define an EntityType which groups the features. As this dataset
    # does not have an ID, a temporary one was created based on the row number.

    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    job_config = bigquery.QueryJobConfig(
        destination=f'{PROJECT_ID}.{DATASET_ID}.train_users'
    )

    query_job = client.query(
        f'''
        SELECT DIV(ROW_NUMBER() OVER(), 100000) user_id, *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        ''',
        job_config=job_config
    )
    query_job.result()

    # Temporary TABLE_ID
    TABLE_ID = 'train_users'

    REGION = 'us-central1'

    BIGQUERY_TABLE = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'
    ID_COLUMN = "user_id"
    IGNORE_COLUMNS_INGESTION = ["user_id", "label"]

    FEATURE_STORE_NAME_PREFIX = "criteo_nvt_e2e"
    FEATURE_STORE_NODE_COUNT = 1

    ENTITY_TYPE_ID = "users"
    ENTITY_TYPE_DESCRIPTION = "Users website navigation"

    IMPORT_WORKER_COUNT = 1

    # Constants based on the params
    BIGQUERY_SOURCE = f"bq://{BIGQUERY_TABLE}"
    API_ENDPOINT = f"{REGION}-aiplatform.googleapis.com"

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    feature_store_path.metadata['bq_source'] = BIGQUERY_SOURCE

    # Create admin_client for CRUD and data_client for reading feature values.
    admin_client = FeaturestoreServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )
    data_client = FeaturestoreOnlineServingServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )

    # Represents featurestore resource path.
    BASE_RESOURCE_PATH = admin_client.common_location_path(PROJECT_ID, REGION)

    FEATURESTORE_ID = f"{FEATURE_STORE_NAME_PREFIX}_{TIMESTAMP}"
    feature_store_path.metadata['featurestore_id'] = FEATURESTORE_ID

    create_lro = admin_client.create_featurestore(
        featurestore_service_pb2.CreateFeaturestoreRequest(
            parent=BASE_RESOURCE_PATH,
            featurestore_id=FEATURESTORE_ID,
            featurestore=featurestore_pb2.Featurestore(
                online_serving_config= \
                    featurestore_pb2.Featurestore.OnlineServingConfig(
                        fixed_node_count=FEATURE_STORE_NODE_COUNT
                ),
            ),
        )
    )
    logging.info(f'Creating feature store {FEATURESTORE_ID}.')
    create_lro.result()

    # Create users entity type. Monitoring disabled.
    users_entity_type_lro = admin_client.create_entity_type(
        featurestore_service_pb2.CreateEntityTypeRequest(
            parent=admin_client.featurestore_path(
                PROJECT_ID, REGION, FEATURESTORE_ID
            ),
            entity_type_id=ENTITY_TYPE_ID,
            entity_type=entity_type_pb2.EntityType(
                description=ENTITY_TYPE_DESCRIPTION
            ),
        )
    )

    # Wait for EntityType creation operation.
    logging.info(f'Creating Entity Type {ENTITY_TYPE_ID}.')
    feature_store_path.metadata['entity_type_id'] = ENTITY_TYPE_ID
    users_entity_type_lro.result()

    create_feature_requests = []
    feature_specs = []
    feature_store_path.metadata['cols_ids_dtype'] = {}

    mapping = {
        'int32': feature_pb2.Feature.ValueType.INT64,
        'hex': feature_pb2.Feature.ValueType.INT64,
    }

    for i, types in enumerate(cols_dtype):
        if columns[i] in IGNORE_COLUMNS_INGESTION:
            continue
        create_feature_requests.append(
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    name=columns[i],
                    value_type=mapping[str(types)],
                    description=columns[i]
                ),
                parent=admin_client.entity_type_path(
                    PROJECT_ID, REGION, FEATURESTORE_ID, ENTITY_TYPE_ID),
                feature_id=re.sub(r'[\W]+', '', columns[i]).lower(),
            )
        )
        feature_specs.append(
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id=re.sub(r'[\W]+', '', columns[i]).lower(), 
                source_field=columns[i]
            )
        )
        feature_store_path.metadata['cols_ids_dtype'][
            re.sub(r'[\W]+', '', columns[i]).lower()
        ] = mapping[str(types)]

    for request in create_feature_requests:
        try:
            logging.info(admin_client.create_feature(request).result())
        except AlreadyExists as e:
            logging.info(e)

    now = time.time()
    seconds = int(now)
    timestamp = Timestamp(seconds=seconds)

    import_request = featurestore_service_pb2.ImportFeatureValuesRequest(
        entity_type=admin_client.entity_type_path(
            PROJECT_ID, REGION, FEATURESTORE_ID, ENTITY_TYPE_ID),
        bigquery_source=io_pb2.BigQuerySource(input_uri=BIGQUERY_SOURCE),
        entity_id_field=ID_COLUMN,
        feature_specs=feature_specs,
        feature_time=timestamp,
        worker_count=IMPORT_WORKER_COUNT,
    )
    ingestion_lro = admin_client.import_feature_values(import_request)
    logging.info('Start to import, will take a couple of minutes.')
    # Polls for the LRO status and prints when the LRO has completed
    # ingestion_lro.result()