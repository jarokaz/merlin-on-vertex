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
    name=config.PREPROCESS_GCS_PIPELINE_NAME
)
def preprocessing_gcs(
    train_paths: list,
    valid_paths: list,
    parquet_output_dir: str,
    sep: str,
    workflow_path: str,
    transformed_output_dir: str,
    shuffle: str,
    recursive: bool
):
    # === Convert CSV to Parquet
    convert_csv_to_parquet = components.convert_csv_to_parquet_op(
        train_paths=train_paths,
        valid_paths=valid_paths,
        output_dir=parquet_output_dir,
        sep=sep
    )
    convert_csv_to_parquet.set_cpu_limit(config.CPU_LIMIT)
    convert_csv_to_parquet.set_memory_limit(config.MEMORY_LIMIT)
    convert_csv_to_parquet.set_gpu_limit(config.GPU_LIMIT)
    convert_csv_to_parquet.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    
    # === Analyze train data split
    analyze_dataset = components.analyze_dataset_op(
        datasets=convert_csv_to_parquet.outputs['output_datasets'],
        workflow_path=workflow_path,
    )
    analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
    analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
    analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
    analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

    # === Transform train data split
    transform_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        transformed_output_dir=transformed_output_dir,
    )
    transform_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    
    # === Transform eval data split
    transform_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        transformed_output_dir=transformed_output_dir,
        split_name='valid'
    )
    transform_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)