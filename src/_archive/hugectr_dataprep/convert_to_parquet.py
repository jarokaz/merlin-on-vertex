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

import argparse
import glob
import logging
import nvtabular as nvt
import numpy as np
import os
import shutil
import time

import utils


CRITEO_FILE_RE = 'day_*'

def convert(args):
    """Converts Criteo TSV files to Parquet."""


   # Create Dask cluster
    client = utils.create_dask_cluster(
        gpus=args.devices.split(','),
        device_memory_fraction=args.device_limit_frac,
        device_pool_fraction=args.device_pool_frac,
        local_directory=args.dask_path,
        protocol=args.protocol
    )

    # Specify column names
    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]
    cols = ["label"] + cont_names + cat_names

    # Specify column dtypes. 
    dtypes = {}
    dtypes["label"] = np.int32
    for x in cont_names:
        dtypes[x] = np.int32
    for x in cat_names:
        dtypes[x] = "hex"

    # Create an NVTabular Dataset from Criteo TSV files 
    file_list = glob.glob(os.path.join(args.input_path, CRITEO_FILE_RE))
    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        names=cols,
        part_mem_fraction=args.part_mem_frac,
        sep="\t",
        dtypes=dtypes,
        client=client,
    )

    # Convert to Parquet
    dataset.to_parquet(
        output_path=args.output_path,
        preserve_files=True,
    )



def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Criteo Preprocessing"))

    parser.add_argument(
        "--input_path", 
        type=str, 
        help="A path to Criteo TSV files")
    parser.add_argument(
        "--output_path", 
        type=str, 
        help="A path to Criteo Parquet files")
    parser.add_argument(
        "--dask_path", 
        type=str, 
        help="A path to Dask working directory") 
    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )
    parser.add_argument(
        "--device_limit_frac",
        default=0.7,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
    )
    parser.add_argument(
        "--device_pool_frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "The RMM pool frac is the same for all GPUs, make sure each one has enough memory size",
    )
    parser.add_argument(
        "--num_io_threads",
        default=0,
        type=int,
        help="Number of threads to use when writing output data (Default 0). "
        "If 0 is specified, multi-threading will not be used for IO.",
    )
    parser.add_argument(
        "--part_mem_frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel(logging.NOTSET)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    args = parse_args()

    convert(args)