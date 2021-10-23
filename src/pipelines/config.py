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


PROJECT_ID = os.getenv("PROJECT_ID", "")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("BUCKET", "")
WORKSPACE = os.getenv("NVT_IMAGE_URI", f"gs://{BUCKET}")

WORKFLOW_PATH = os.getenv(
    "WORKFLOW_PATH", 
    os.path.join(WORKSPACE, 'workflow')
) # Location of the workflow artifact.
PARQUET_OUTPUT_DIR = os.getenv(
    "PARQUET_OUTPUT_DIR", 
    os.path.join(WORKSPACE, 'parquet_raw_data')
) # Location of the converted Parquet data files.
TRANSFORMED_OUTPUT_DIR = os.getenv(
    "TRANSFORMED_OUTPUT_DIR",
    os.path.join(WORKSPACE, 'transformed_data')
) # Location of the transformed data files.

BQ_DATASET_NAME = os.getenv("BQ_DATASET_NAME", "criteo")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")
BQ_TRAIN_TABLE_NAME = os.getenv("BQ_TRAIN_TABLE_NAME", "train")
BQ_VALID_TABLE_NAME = os.getenv("BQ_VALID_TABLE_NAME", "valid")

PREPROCESS_CSV_PIPELINE_NAME = os.getenv("PREPROCESS_CSV_PIPELINE_NAME", "nvt-csv-pipeline")
PREPROCESS_BQ_PIPELINE_NAME = os.getenv("PREPROCESS_BQ_PIPELINE_NAME", "nvt-bq-pipeline")
TRAINING_PIPELINE_NAME = os.getenv("TRAINING_PIPELINE_NAME", "merlin-e2e-pipeline")

MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "criteo-merlin-recommender")

NVT_IMAGE_URI = os.getenv("NVT_IMAGE_URI",  f"gcr.io/{PROJECT_ID}/nvt_preprocessing")
HUGECTR_IMAGE_URI = os.getenv("HUGECTR_IMAGE_URI",  f"gcr.io/{PROJECT_ID}/hugectr_training")

MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "120G")
CPU_LIMIT = os.getenv("CPU_LIMIT", "32")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "nvidia-tesla-t4")

MACHINE_TYPE = os.getenv("MACHINE_TYPE", "a2-highgpu-4g")
ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", "NVIDIA_TESLA_A100")
ACCELERATOR_NUM = os.getenv("ACCELERATOR_NUM", "4")
NUM_WORKERS = os.getenv("NUM_WORKERS", "4")
