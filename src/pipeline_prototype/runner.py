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

import argparse
import json
import kfp
import logging

from typing import NamedTuple
import numpy as np

from google.cloud import aiplatform

from kfp.v2 import compiler
from pipeline import preprocessing_pipeline


PACKAGE_PATH = 'nvt_pipeline.json'


def compile_pipeline(package_path):
    compiler.Compiler().compile(
       pipeline_func=preprocessing_pipeline,
       package_path=package_path)
    
    
def run_pipeline(package_path, parameter_values, vertex_sa):
    
    pipeline_job = aiplatform.PipelineJob(
        display_name=JOB_NAME,
        template_path=PACKAGE_PATH,
        enable_caching=True,
        parameter_values=parameter_values,
    )
    
    pipeline_job.run(
        service_account=vertex_sa,
        sync=False
     )
    
def define_schema():
    
    cont_features = [[name, "int32"] for name in ["I" + str(x) for x in range(1, 14)]]
    cat_features = [[name, "hex"] for name in ["C" + str(x) for x in range(1, 27)]]
    schema = [['label', 'int32']] + cont_features + cat_features
    
    return schema
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        default='jk-mlops-dev',
                        help='Project ID')
    parser.add_argument('--region',
                        type=str,
                        default='us-central1',
                        help='Region')
    parser.add_argument('--gcs_bucket',
                        type=str,
                        default='gs://jk-vertex-us-central1',
                        help='GCS bucket')
    parser.add_argument('--vertex_sa',
                        type=str,
                        default='vertex-sa@jk-mlops-dev.iam.gserviceaccount.com',
                        help='Vertex SA')
    
    parser.add_argument('--train_files',
                        type=str,
                        default='/gcs/jk-criteo-bucket/criteo_orig/day_0',
                        help='List of training files')
    parser.add_argument('--valid_files',
                        type=str,
                        default='/gcs/jk-criteo-bucket/criteo_orig/day_1',
                        help='List of validation files')
    parser.add_argument('--sep',
                        type=str,
                        default='\t',
                        help='Field separator')
    parser.add_argument('--gpus',
                        type=str,
                        default='0,1',
                        help='GPU devices to use for Preprocessing')
    
    
    parser.add_argument('--machine_type',
                        type=str,
                        default='a2-highgpu-2g',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='NVIDIA_TESLA_A100',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        default=2,
                        help='Num of GPUs')


    parser.add_argument('--part_mem_frac',
                        type=float,
                        required=False,
                        default=0.10,
                        help='Desired maximum size of each partition as a fraction of total GPU memory')
    parser.add_argument('--device_limit_frac',
                        type=float,
                        required=False,
                        default=0.7,
                        help='Device limit fraction')
    parser.add_argument('--device_pool_frac',
                        type=float,
                        required=False,
                        default=0.7,
                        help='Device pool fraction')

    args = parser.parse_args()
    
    args.train_files = args.train_files.split(',')
    args.valid_files = args.valid_files.split(',')

    
    return args

    
if __name__ == '__main__':


    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    
    args = parse_args()

    logging.info(f"Args: {args}")
    

    params = {
        'train_files': json.dumps(args.train_files),
        'valid_files': json.dumps(args.valid_files),
        'sep': args.sep,
        'schema': json.dumps(define_schema()),
        'gpus': json.dumps(args.gpus.split(','))
    }
    
    compile_pipeline(PACKAGE_PATH)
    
    run_pipeline(PACKAGE_PATH, params, args.vertex_sa)
    
    