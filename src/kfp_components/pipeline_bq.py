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

from typing import Optional
import kfp_components
from kfp.v2 import dsl

PIPELINE_NAME = 'nvt-pipeline-bq'


@dsl.pipeline(
    name=PIPELINE_NAME
)
def preprocessing_pipeline_bq(
    bq_table_train: str,
    bq_table_valid: str,
    output_path: str,
    bq_project: str,
    bq_dataset_id: str,
    location: str,
    gpus: str,
    workflow_path: str,
    output_transformed: str,
    shuffle: str,
    recursive: bool
):
    # === Convert CSV to Parquet
    export_parquet_from_bq = kfp_components.export_parquet_from_bq_op(
        bq_table_train=bq_table_train,
        bq_table_valid=bq_table_valid,
        output_path=output_path,
        bq_project=bq_project,
        bq_dataset_id=bq_dataset_id,
        location=location
    )
    export_parquet_from_bq.set_cpu_limit("8")
    export_parquet_from_bq.set_memory_limit("32G")
    
    # === Fit dataset
    fit_dataset = kfp_components.fit_dataset_op(
        datasets=export_parquet_from_bq.outputs['output_datasets'],
        workflow_path=workflow_path,
        gpus=gpus
    )
    fit_dataset.set_cpu_limit("8")
    fit_dataset.set_memory_limit("32G")
    fit_dataset.set_gpu_limit("1")
    fit_dataset.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')

    # === Transform dataset
    transform_dataset = kfp_components.transform_dataset_op(
        fitted_workflow=fit_dataset.outputs['fitted_workflow'],
        output_transformed=output_transformed,
        gpus=gpus
    )
    transform_dataset.set_cpu_limit("8")
    transform_dataset.set_memory_limit("32G")
    transform_dataset.set_gpu_limit("1")
    transform_dataset.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')