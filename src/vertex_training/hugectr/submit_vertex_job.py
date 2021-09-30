# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Submitting HugeCTR training jobs to Vertex AI Training."""

import argparse
import json
import logging
import pprint
import time

from google.cloud import aiplatform


TRAINING_MODULE = 'trainer.train'


def submit_job(args):
    """Submits a Vertex Training Custom Job."""

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.gcs_bucket
    )

    batchsize = args.per_gpu_batchsize * args.accelerator_num
    gpus=json.dumps([list(range(args.accelerator_num))]).replace(' ','')

    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_num,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.train_image,
                "command": ["python", "train.py"],
                "args": [
                    '--batchsize=' + str(batchsize),
                    '--train_data=' + args.train_data, 
                    '--valid_data=' + args.valid_data,
                    '--slot_size_array=' + args.slot_size_array,
                    '--max_iter=' + str(args.max_iter),
                    '--num_epochs=' + str(args.num_epochs),
                    '--eval_interval=' + str(args.eval_interval),
                    '--snapshot=' + str(args.snapshot),
                    '--display_interval=' + str(args.display_interval),
                    '--workspace_size_per_gpu=' + str(args.workspace_size_per_gpu),
                    '--gpus=' + gpus,
                ],
            },
        }
    ]


    job_name = 'HUGECTR_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    logging.info(f'Starting job: {job_name}')

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
    )
    job.run(sync=True,
            restart_job_on_worker_restart=False
    )

def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--project',
                        type=str,
                        required=True,
                        help='Project ID')
    parser.add_argument('--region',
                        type=str,
                        required=True,
                        help='Region')
    parser.add_argument('--gcs_bucket',
                        type=str,
                        required=True,
                        help='GCS staging bucket')
    parser.add_argument('--vertex_sa',
                        type=str,
                        required=True,
                        help='Vertex Training service account')
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
    parser.add_argument('--train_image',
                        type=str,
                        required=True,
                        help='Training image name')

    parser.add_argument('--train_data',
                        type=str,
                        required=True,
                        help='A GCS location of the training data _file_list.txt')
    parser.add_argument('--valid_data',
                        type=str,
                        help='A GCS location of the training data _file_list.txt')
    parser.add_argument('--max_iter',
                        type=int,
                        default=0,
                        help='A number of training iterations')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='A number of training epochs')
    parser.add_argument('--per_gpu_batchsize',
                        type=int,
                        default=2048,
                        help='Per GPU batch size')
    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=0,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('--slot_size_array',
                        type=str,
                        required=False,
                        default='16961592,34319,16768,7378,20132,4,6955,1384,63,11492137,914596,289081,11,2209,10737,79,4,971,15,17618173,5049264,15182940,364088,12075,102,35',
                        help='Categorical variables cardinalities')
    parser.add_argument('--eval_interval',
                        type=int,
                        required=False,
                        default=5000,
                        help='Run evaluation after given number of iterations')
    parser.add_argument('--display_interval',
                        type=int,
                        required=False,
                        default=1000,
                        help='Display progress after given number of iterations')
    parser.add_argument('--workspace_size_per_gpu',
                        type=int,
                        required=False,
                        default=61,
                        help='Workspace size per gpu in MB')
    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    return args  

if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")

    submit_job(args)