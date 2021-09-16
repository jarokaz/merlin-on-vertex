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

import components

from kfp.v2 import dsl

PIPELINE_NAME = 'nvt-test-pipeline'


@dsl.pipeline(
    name=PIPELINE_NAME
)
def preprocessing_pipeline(
    train_files: list,
    valid_files: list,
    sep: str,
    gpus: list,
    schema: list,
):
    ingest_csv_files = components.ingest_csv_op(
        train_files=train_files,
        valid_files=valid_files,
        sep=sep,
        gpus=gpus,
        schema=schema,
    )
    ingest_csv_files.set_cpu_limit("48")
    ingest_csv_files.set_memory_limit("312G")
    ingest_csv_files.set_gpu_limit("4")
    ingest_csv_files.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')
    
    fit_workflow = components.fit_workflow_op(
        dataset=ingest_csv_files.outputs['output_dataset'],
        gpus=gpus
    )
    fit_workflow.set_cpu_limit("48")
    fit_workflow.set_memory_limit("312G")
    fit_workflow.set_gpu_limit("4")
    fit_workflow.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')