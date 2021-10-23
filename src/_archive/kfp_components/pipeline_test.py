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

from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input, 
    InputPath, 
    Model, 
    Output,
    OutputPath
)


PIPELINE_NAME = 'test-nvt-pipeline-gcs-feat'


@dsl.pipeline(
    name=PIPELINE_NAME
)
def test_pipeline_gcs_feat(
    cols_dtype: dict
):
    # === Create feature store and load data
    load_bq_to_feature_store = \
        kfp_components.load_bq_to_feature_store_op(
            cols_dtype = cols_dtype
    )
    load_bq_to_feature_store.set_cpu_limit("8")
    load_bq_to_feature_store.set_memory_limit("32G")