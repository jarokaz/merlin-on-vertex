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
"""Preprocessing pipeline prototype."""

from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath)
from typing import Optional

BASE_IMAGE_NAME = f'gcr.io/jk-mlops-dev/nvt_base_image'

@dsl.component(base_image=BASE_IMAGE_NAME)
def ingest_csv_op(
    train_files: list,
    valid_files: list,
    sep: str,
    schema: list,
    gpus: list,
    output_dataset: Output[Dataset]
):
    import logging
    import nvtabular as nvt
    import numpy as np
    import os
    
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    
    TRAIN_SPLIT_FOLDER = 'train'
    VALID_SPLIT_FOLDER = 'valid'
    
    client = None
    if len(gpus) > 1:
        logging.info('Creating a Dask CUDA cluster')
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=','.join(gpus),
            n_workers=len(gpus)
        )
        client = Client(cluster)
    
    names = [feature[0] for feature in schema]
    dtypes = {feature[0]: feature[1] for feature in schema}
    
    for folder_name, files in zip([TRAIN_SPLIT_FOLDER, VALID_SPLIT_FOLDER], [train_files, valid_files]):
        dataset = nvt.Dataset(
            path_or_source = files,
            engine='csv',
            names=names,
            sep=sep,
            dtypes=dtypes,
            client=client
        )
        
        output_path = os.path.join(output_dataset.uri, folder_name)
        os.makedirs(output_path, exist_ok=True)
        
        logging.info('Writing a parquet file to {}'.format(output_path))
        dataset.to_parquet(
            output_path=output_path,
            preserve_files=True
        )
    
    output_dataset.metadata['split_names'] = [TRAIN_SPLIT_FOLDER, VALID_SPLIT_FOLDER]
    
    
@dsl.component(base_image=BASE_IMAGE_NAME)
def fit_workflow_op(
    dataset: Input[Dataset],
    fitted_workflow: Output[Artifact],
    gpus: list,
    part_mem_frac: Optional[float]=0.1,
    device_limit_frac: Optional[float]=0.7,
    device_pool_frac: Optional[float]=0.8,
    split_name: Optional[str]='train'
):
    import logging
    import nvtabular as nvt
    import numpy as np
    
    from pathlib import Path
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from nvtabular.utils import _pynvml_mem_size, device_mem_size
    
    from nvtabular.ops import (
        Categorify,
        Clip,
        FillMissing,
        Normalize,
    )
    
    STATS_FOLDER = 'stats'
    WORKFLOW_FOLDER = 'workflow'
    
    if not split_name in dataset.metadata['split_names']:
        raise RuntimeError('Dataset does not have {} split'.format(split_name))
        
 
    CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
    CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
    LABEL_COLUMNS = ["label"]
    COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMNS

    
    device_size = device_mem_size(kind="total")
    part_size = int(part_mem_frac * device_size)
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    
    client = None
    if len(gpus) > 1:
        logging.info('Creating a Dask CUDA cluster')
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=','.join(gpus),
            n_workers=len(gpus),
            device_memory_limit=device_limit,
            rmm_pool_size=(device_pool_size // 256) * 256
        )
        client = Client(cluster)
    
    num_buckets = 10000000
    cat_features = CATEGORICAL_COLUMNS >> categorify_op
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + LABEL_COLUMNS

    workflow = nvt.Workflow(features, client=client)  
    
    train_paths = [str(path) for path in Path(dataset.uri, split_name).glob('*.parquet')]
    train_dataset = nvt.Dataset(train_paths, engine="parquet", part_size=part_size)
    
    workflow.fit(train_dataset)
    workflow.save(fitted_workflow.uri)