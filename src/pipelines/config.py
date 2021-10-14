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
"""Vertex pipeline configurations."""

import os

IMAGE_URI = os.getenv("IMAGE_URI", "")

PREPROCESS_GCS_PIPELINE_NAME = os.getenv("PREPROCESS_GCS_PIPELINE_NAME", "nvt-gcs-pipeline")
PREPROCESS_BQ_PIPELINE_NAME = os.getenv("PREPROCESS_BQ_PIPELINE_NAME", "nvt-bq-pipeline")
PREPROCESS_E2E_PIPELINE_NAME = os.getenv("PREPROCESS_E2E_PIPELINE_NAME", "merlin-e2e-pipeline")

MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "120G")
CPU_LIMIT = os.getenv("CPU_LIMIT", "32")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "nvidia-tesla-t4")