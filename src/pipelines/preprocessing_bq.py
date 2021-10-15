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

"""Preprocessing pipeline"""

from . import components
from kfp.v2 import dsl
from . import config

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'


@dsl.pipeline(
    name=config.PREPROCESS_BQ_PIPELINE_NAME
)
def preprocessing_bq(
    bq_table_train: str,
    bq_table_valid: str,
    output_dir: str,
    bq_project: str,
    bq_dataset_id: str,
    location: str,
    workflow_path: str,
    transformed_output_dir: str,
    shuffle: str,
    recursive: bool
):
    # === Export Bigquery tables as PARQUET files
    export_parquet_from_bq = components.export_parquet_from_bq_op(
        bq_table_train=bq_table_train,
        bq_table_valid=bq_table_valid,
        output_dir=output_dir,
        bq_project=bq_project,
        bq_dataset_id=bq_dataset_id,
        location=location
    )
    export_parquet_from_bq.set_cpu_limit(config.CPU_LIMIT)
    export_parquet_from_bq.set_memory_limit(config.MEMORY_LIMIT)

    # === Analyze train data split
    analyze_dataset = components.analyze_dataset_op(
        datasets=export_parquet_from_bq.outputs['output_datasets'],
        workflow_path=workflow_path,
        n_workers=int(config.GPU_LIMIT)
    )
    analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
    analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
    analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
    analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

    # === Transform train data split
    transform_train_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        transformed_output_dir=transformed_output_dir,
        n_workers=int(config.GPU_LIMIT)
    )
    transform_train_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_train_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_train_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_train_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    
    # === Transform eval data split
    transform_valid_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        transformed_output_dir=transformed_output_dir,
        split_name='valid',
        n_workers=int(config.GPU_LIMIT)
    )
    transform_valid_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_valid_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_valid_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_valid_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
