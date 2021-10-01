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

import os
import rmm

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from nvtabular import utils as nvt_utils

    
def create_dask_cluster(
    gpus,
    device_memory_fraction,
    device_pool_fraction,
    local_directory,
    protocol="tcp",
):
    """Deploys a LocalCUDACluster DASK cluster."""

    device_memory = nvt_utils.device_mem_size()
    device_pool_size = device_memory * device_pool_fraction
    device_memory_limit = device_memory * device_memory_fraction

    def _rmm_pool():
        rmm.reinitialize(
            # RMM may require the pool size to be a multiple of 256.
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,
        ) 

    if protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS
        cluster = LocalCUDACluster(
            protocol=protocol,
            CUDA_VISIBLE_DEVICES=','.join(gpus),
            n_workers=len(gpus),
            enable_nvlink=True,
            device_memory_limit=device_memory_limit,
            local_directory=local_directory
            )
    else:
        cluster = LocalCUDACluster(
            protocol=protocol,
            n_workers=len(gpus),
            CUDA_VISIBLE_DEVICES=','.join(gpus),
            device_memory_limit=device_memory_limit,
            local_directory=local_directory,
            )
        client = Client(cluster)

    if device_pool_size > 256:
        client.run(_rmm_pool)

    return client


def bytesto(bytes, to, bsize=1024):
    """Computes a partition size in GBs."""
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])