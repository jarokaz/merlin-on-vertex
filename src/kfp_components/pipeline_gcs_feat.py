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

import kfp_components
from kfp.v2 import dsl

PIPELINE_NAME = 'nvt-pipeline-gcs-feat'


@dsl.pipeline(
    name=PIPELINE_NAME
)
def preprocessing_pipeline_gcs_feat(
    train_paths: list,
    valid_paths: list,
    output_path: str,
    columns: list,
    cols_dtype: dict,
    sep: str,
    gpus: str,
    workflow_path: str,
    output_transformed: str,
    shuffle: str,
    recursive: bool,
    bq_project: str,
    bq_dataset_id: str,
    bq_dest_table_id: str
):
    # === Convert CSV to Parquet
    convert_csv_to_parquet = kfp_components.convert_csv_to_parquet_op(
        train_paths=train_paths,
        valid_paths=valid_paths,
        output_path=output_path,
        columns=columns,
        cols_dtype=cols_dtype,
        sep=sep,
        gpus=gpus,
    )
    convert_csv_to_parquet.set_cpu_limit("8")
    convert_csv_to_parquet.set_memory_limit("32G")
    convert_csv_to_parquet.set_gpu_limit("1")
    convert_csv_to_parquet.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')
    
    # === Fit dataset
    fit_dataset = kfp_components.fit_dataset_op(
        datasets=convert_csv_to_parquet.outputs['output_datasets'],
        workflow_path=workflow_path,
        gpus=gpus
    )
    fit_dataset.set_cpu_limit("8")
    fit_dataset.set_memory_limit("32G")
    fit_dataset.set_gpu_limit("1")
    fit_dataset.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')

    # === Transform dataset
    transformed_dataset = kfp_components.transform_dataset_op(
        fitted_workflow=fit_dataset.outputs['fitted_workflow'],
        output_transformed=output_transformed,
        gpus=gpus
    )
    transformed_dataset.set_cpu_limit("8")
    transformed_dataset.set_memory_limit("32G")
    transformed_dataset.set_gpu_limit("1")
    transformed_dataset.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-t4')

    # === Load parquet file to BigQuery
    import_parquet_to_bq = kfp_components.import_parquet_to_bq_op(
        transformed_dataset = transformed_dataset.outputs['transformed_dataset'],
        bq_project = bq_project,
        bq_dataset_id = bq_dataset_id,
        bq_dest_table_id = bq_dest_table_id
    )
    import_parquet_to_bq.set_cpu_limit("8")
    import_parquet_to_bq.set_memory_limit("32G")

    # === Create feature store and load data
    load_bq_to_feature_store = kfp_components.load_bq_to_feature_store_op(
        output_bq_table = \
            import_parquet_to_bq.outputs['output_bq_table'],
        cols_dtype = cols_dtype
    )
    load_bq_to_feature_store.set_cpu_limit("8")
    load_bq_to_feature_store.set_memory_limit("32G")