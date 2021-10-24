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
"""Preprocessing pipelines."""

from . import components
from kfp.v2 import dsl
from . import config

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'

@dsl.pipeline(
    name=config.PREPROCESS_CSV_PIPELINE_NAME
)
def preprocessing_csv(
    train_paths: list,
    valid_paths: list,
    parquet_output_dir: str,
    workflow_path: str,
    transformed_output_dir: str,
    sep: str,
    shuffle: str
):
    # ==================== Convert from CSV to Parquet ========================

    # === Convert train dataset from CSV to Parquet
    csv_to_parquet_train = components.convert_csv_to_parquet_op(
        data_paths=train_paths,
        split='train',
        output_dir=parquet_output_dir,
        sep=sep,
        n_workers=int(config.GPU_LIMIT)
    )
    csv_to_parquet_train.set_cpu_limit(config.CPU_LIMIT)
    csv_to_parquet_train.set_memory_limit(config.MEMORY_LIMIT)
    csv_to_parquet_train.set_gpu_limit(config.GPU_LIMIT)
    csv_to_parquet_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    
    # === Convert eval dataset from CSV to Parquet
    csv_to_parquet_valid = components.convert_csv_to_parquet_op(
        data_paths=valid_paths,
        split='valid',
        output_dir=parquet_output_dir,
        sep=sep,
        n_workers=int(config.GPU_LIMIT)
    )
    csv_to_parquet_valid.set_cpu_limit(config.CPU_LIMIT)
    csv_to_parquet_valid.set_memory_limit(config.MEMORY_LIMIT)
    csv_to_parquet_valid.set_gpu_limit(config.GPU_LIMIT)
    csv_to_parquet_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

    # ==================== Analyse train dataset ==============================

    # === Analyze train data split
    analyze_dataset = components.analyze_dataset_op(
        parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
        workflow_path=workflow_path,
        n_workers=int(config.GPU_LIMIT)
    )
    analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
    analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
    analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
    analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

    # ==================== Transform train and validation dataset =============

    # === Transform train data split
    transform_train_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
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
        parquet_dataset=csv_to_parquet_valid.outputs['output_dataset'],
        transformed_output_dir=transformed_output_dir,
        n_workers=int(config.GPU_LIMIT)
    )
    transform_valid_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_valid_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_valid_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_valid_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)


@dsl.pipeline(
    name=config.PREPROCESS_BQ_PIPELINE_NAME
)
def preprocessing_bq(
    bq_project: str,
    bq_dataset_name: str,
    bq_train_table_name: str,
    bq_valid_table_name: str,
    bq_location: str,
    parquet_output_dir: str,
    workflow_path: str,
    transformed_output_dir: str,
    shuffle: str
):
    # ==================== Exporting tables as Parquet ========================

    # === Export train table as parquet
    export_train_from_bq = components.export_parquet_from_bq_op(
        bq_project=bq_project,
        bq_dataset_name=bq_dataset_name,
        bq_location=bq_location,
        bq_table_name=bq_train_table_name,
        split='train',
        output_dir=parquet_output_dir,
    )
    export_train_from_bq.set_cpu_limit(config.CPU_LIMIT)
    export_train_from_bq.set_memory_limit(config.MEMORY_LIMIT)

    # === Export valid table as parquet
    export_valid_from_bq = components.export_parquet_from_bq_op(
        bq_project=bq_project,
        bq_dataset_name=bq_dataset_name,
        bq_location=bq_location,
        bq_table_name=bq_valid_table_name,
        split='valid',
        output_dir=parquet_output_dir,
    )
    export_valid_from_bq.set_cpu_limit(config.CPU_LIMIT)
    export_valid_from_bq.set_memory_limit(config.MEMORY_LIMIT)

    # ==================== Analyse train dataset ==============================

    # === Analyze train data split
    analyze_dataset = components.analyze_dataset_op(
        parquet_dataset=export_train_from_bq.outputs['output_dataset'],
        workflow_path=workflow_path,
        n_workers=int(config.GPU_LIMIT)
    )
    analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
    analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
    analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
    analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

    # ==================== Transform train and validation dataset =============

    # === Transform train data split
    transform_train_dataset = components.transform_dataset_op(
        workflow=analyze_dataset.outputs['workflow'],
        parquet_dataset=export_train_from_bq.outputs['output_dataset'],
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
        parquet_dataset=export_valid_from_bq.outputs['output_dataset'],
        transformed_output_dir=transformed_output_dir,
        n_workers=int(config.GPU_LIMIT)
    )
    transform_valid_dataset.set_cpu_limit(config.CPU_LIMIT)
    transform_valid_dataset.set_memory_limit(config.MEMORY_LIMIT)
    transform_valid_dataset.set_gpu_limit(config.GPU_LIMIT)
    transform_valid_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
