# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Standard Libraries
import argparse
import logging
import pprint
import time

from google.cloud import aiplatform


PREPROCESS_FILE = 'dask-nvtabular-criteo-benchmark.py'


def run(args):

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.gcs_bucket
    )

    job_name = 'NVT_BENCHMARK_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_num,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.preprocess_image,
                "command": ["python", PREPROCESS_FILE],
                "args": [
                    '--data-path=' + args.data_path, 
                    '--out-path=' + f'{args.out_path}/{job_name}',
                    '--devices=' + args.devices,
                    '--protocol=' + args.protocol, 
                    '--device-limit-frac=' + str(args.device_limit_frac), 
                    '--device-pool-frac=' + str(args.device_pool_frac), 
                    '--part-mem-frac=' + str(args.part_mem_frac),
                    '--num-io-threads=' + str(args.num_io_threads),
                ],
            },
        }
    ]

    logging.info(f'Starting job: {job_name}')

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        
    )
    job.run(sync=True,
            restart_job_on_worker_restart=False,
    )

if __name__ == '__main__':

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
                        default='training-sa@jk-mlops-dev.iam.gserviceaccount.com',
                        help='Vertex SA')
    parser.add_argument('--machine_type',
                        type=str,
                        default='n1-standard-96',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='NVIDIA_TESLA_T4',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        default=4,
                        help='Num of GPUs')
    parser.add_argument('--preprocess_image',
                        type=str,
                        default='gcr.io/jk-mlops-dev/nvt-test',
                        help='Training image name')

    parser.add_argument('---data-path',
                        type=str,
                        default='/gcs/jk-criteo-bucket/criteo_1_per_file',
                        help='Criteo parquet data location')
    parser.add_argument('----out-path',
                        type=str,
                        default='/gcs/jk-criteo-bucket',
                        help='Output GCS location')
    parser.add_argument("--protocol",
                        choices=["tcp", "ucx"],
                        default="tcp",
                        type=str,
                        help="Communication protocol to use (Default 'tcp')")
    parser.add_argument('--part_mem_frac',
                        type=float,
                        required=False,
                        default=0.12,
                        help='Desired maximum size of each partition as a fraction of total GPU memory')
    parser.add_argument('--device_limit_frac',
                        type=float,
                        required=False,
                        default=0.8,
                        help='Device limit fraction')
    parser.add_argument('--device_pool_frac',
                        type=float,
                        required=False,
                        default=0.9,
                        help='Device pool fraction')
    parser.add_argument("--num-io-threads",
                        default=0,
                        type=int,
                        help="Number of threads to use when writing output data (Default 0). "
                        "If 0 is specified, multi-threading will not be used for IO.")
    parser.add_argument("--devices",
                        default="0,1,3,4",
                        type=str,
                        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
                        "The number of visible devices dictates the number of Dask workers (GPU processes) "
                        "The CUDA_VISIBLE_DEVICES environment variable will be used by default")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run(args)