from typing import Dict, Union

import numpy as np
import fsspec
import os

import nvtabular as nvt
from nvtabular.utils import device_mem_size, get_rmm_size
from nvtabular.io.shuffle import Shuffle
from nvtabular.ops import (
    Categorify,
    Clip,
    FillMissing,
    Normalize,
)

# External Dependencies
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from google.cloud import bigquery


def get_criteo_col_dtypes() -> Dict[str,Union[str, np.int32]]:
    # Specify column dtypes. Note that "hex" means that
    # the values will be hexadecimal strings that should
    # be converted to int32
    col_dtypes = {}

    col_dtypes["label"] = np.int32
    for x in ["I" + str(i) for i in range(1, 14)]:
        col_dtypes[x] = np.int32
    for x in ["C" + str(i) for i in range(1, 27)]:
        col_dtypes[x] = 'hex'

    return col_dtypes


def create_convert_cluster() -> Client:
    cluster = LocalCUDACluster(
        rmm_pool_size=get_rmm_size(0.8 * device_mem_size())
    )
    return Client(cluster)


def create_csv_dateset(
    data_paths: list, 
    sep: str,
    recursive: bool,
    col_dtypes: Dict[str,str],
    client: Client
) -> nvt.Dataset:
    fs_spec = fsspec.filesystem('gs')
    rec_symbol = '**' if recursive else '*'

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
            print('Incorrect path: {path}.')
        except OSError as os_err:
            print(os_err)
            print(f'Verify access to the bucket.')

    return nvt.Dataset(
        path_or_source = valid_paths,
        engine='csv',
        names=list(col_dtypes.keys()),
        sep=sep,
        dtypes=col_dtypes,
        client=client,
        assume_missing=True
    )


def convert_csv_to_parquet(
    output_path: str,
    dataset: nvt.Dataset,
    shuffle: nvt.io.Shuffle = None
):
    if shuffle:
        shuffle = getattr(Shuffle, shuffle)

    dataset.to_parquet(
        output_path,
        preserve_files=True,
        shuffle=shuffle
    )


def create_criteo_nvt_workflow() -> nvt.Workflow:
    # Columns definition
    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]

    # Transformation pipeline
    num_buckets = 10000000
    categorify_op = Categorify(max_size=num_buckets)
    cat_features = cat_names >> categorify_op
    cont_features = cont_names >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + ['label']

    # Create and save workflow
    return nvt.Workflow(features)


def create_transform_cluster(
    device_limit_frac: float,
    device_pool_frac: float,
) -> Client:
    device_size = device_mem_size()
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    rmm_pool_size = (device_pool_size // 256) * 256

    cluster = LocalCUDACluster(
        device_memory_limit=device_limit,
        rmm_pool_size=rmm_pool_size
    )
    
    return Client(cluster)


def create_fit_dataset(
    data_path: str,
    part_mem_frac: float,
    client
) -> nvt.Dataset:
    return nvt.Dataset(
        os.path.join(data_path, '*.parquet'),
        engine="parquet", 
        part_size=int(part_mem_frac * device_mem_size()),
        client=client
    )


def fit_and_save_workflow(
    workflow: nvt.Workflow, 
    dataset: nvt.Dataset, 
    workflow_path: str
):
    workflow.fit(dataset)
    workflow.save(workflow_path)


def workflow_transform(
    data_path: str,
    part_mem_frac: float,
    client: Client,
    workflow_path: str,
    destination_transformed: str,
    shuffle: str = None
):
    if shuffle:
        shuffle = getattr(Shuffle, shuffle)

    # Load Transformation steps
    workflow = nvt.Workflow.load(workflow_path, client)

    dataset = nvt.Dataset(
        os.path.join(data_path, '*.parquet'),
        engine="parquet", 
        part_size=int(part_mem_frac * device_mem_size())
    )

    workflow.transform(dataset).to_parquet(
        output_path=destination_transformed,
        shuffle=shuffle
    )

def extract_table_from_bq(
    client,
    output_converted,
    folder_name,
    dataset_ref,
    table_id,
    location = 'us'
):
    extract_job_config = bigquery.ExtractJobConfig()
    extract_job_config.destination_format = 'PARQUET'

    bq_glob_path = os.path.join(
        'gs://', 
        output_converted,
        folder_name,
        f'{folder_name}-*.parquet'
    )
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref, 
        bq_glob_path, 
        location=location,
        job_config=extract_job_config
    )
    extract_job.result()