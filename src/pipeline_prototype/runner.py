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


import kfp
import json

from typing import NamedTuple
import numpy as np

from google.cloud import aiplatform

from kfp.v2 import compiler
from pipeline import preprocessing_pipeline

PROJECT_ID = 'jk-mlops-dev'
REGION = 'us-central1'
STAGING_BUCKET = 'gs://jk-vertex-us-central1'
VERTEX_SA = f'vertex-sa@{PROJECT_ID}.iam.gserviceaccount.com'
JOB_NAME = 'test_pipeline_run'
PACKAGE_PATH = 'nvt_pipeline.json'

compiler.Compiler().compile(
    pipeline_func=preprocessing_pipeline,
    package_path=PACKAGE_PATH)


cont_features = [[name, "int32"] for name in ["I" + str(x) for x in range(1, 14)]]
cat_features = [[name, "hex"] for name in ["C" + str(x) for x in range(1, 27)]]
schema = [['label', 'int32']] + cont_features + cat_features
sep = "\t"
gpus = ['0','1']

train_files = ['/gcs/jk-criteo-bucket/criteo_orig/day_0']
valid_files = ['/gcs/jk-criteo-bucket/criteo_orig/day_1']

params = {
    'train_files': json.dumps(train_files),
    'valid_files': json.dumps(valid_files),
    'sep': "\t",
    'schema': json.dumps(schema),
    'gpus': json.dumps(gpus)
}

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=STAGING_BUCKET
)

pipeline_job = aiplatform.PipelineJob(
    display_name=JOB_NAME,
    template_path=PACKAGE_PATH,
    enable_caching=False,
    parameter_values=params,
)

pipeline_job.run(
    service_account=VERTEX_SA,
    sync=False
)


