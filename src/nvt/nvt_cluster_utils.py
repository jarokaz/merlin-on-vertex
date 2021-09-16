# Standard Libraries
import os
import re
import shutil
import warnings
import glob
from pathlib import Path

# External Dependencies
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm
from google.cloud import storage

# NVTabular
from nvtabular.utils import _pynvml_mem_size, device_mem_size
from nvtabular.utils import get_rmm_size


class NvtClusterUtils:
    def __init__(self,
                 local_base_dir: str,
                 visible_devices: str,
                 protocol: str = 'tcp',
                 device_limit_frac: float = 0.8,
                 device_pool_frac: float = 0.9,
                 part_mem_frac: float = 0.125,
                 dashboard_port: str = '8787'):
        
        self.local_base_dir = Path(local_base_dir)
        self.dashboard_port = dashboard_port

        self.protocol = protocol
        self.visible_devices = visible_devices # '0,1,2,3'
        self.n_workers=len(visible_devices.split(','))

        self.device_size = device_mem_size()
        self.device_limit = int(device_limit_frac * self.device_size)
        self.device_pool_size = int(device_pool_frac * self.device_size)
        self.part_size = int(part_mem_frac * self.device_size)

        self.rmm_pool_size = (self.device_pool_size // 256) * 256

    def initialize_cluster(self, port: int = None) -> Client:
        self.check_device_mem_occupancy()
        cluster = port  # (Optional) Specify existing scheduler port
        if cluster is None:
            cluster = LocalCUDACluster(
                protocol=self.protocol,
                n_workers=len(self.visible_devices.split(",")),
                CUDA_VISIBLE_DEVICES=self.visible_devices,
                device_memory_limit=self.device_limit,
                local_directory=str(self.local_base_dir),
                dashboard_address=":" + self.dashboard_port,
                rmm_pool_size=self.rmm_pool_size
            )
        client = Client(cluster)

        return client

    def check_device_mem_occupancy(self):
        # Check if any device memory is already occupied
        for dev in self.visible_devices.split(","):
            fmem = _pynvml_mem_size(kind="free", index=int(dev))
            used = (self.device_size - fmem) / 1e9
            if used > 1.0:
                warnings.warn(f'BEWARE - {used} GB is \
                                already occupied on device {int(dev)}!')

    def setup_rmm_pool(self, client: Client):
        ''' Initialize an RMM pool allocator.
        Note: RMM may require the pool size to be a multiple of 256.'''
        pool_size = get_rmm_size(self.device_pool_size)
        client.run(rmm.reinitialize,
                   pool_allocator=True, 
                   initial_pool_size=pool_size)
            
    def setup_local_dask_dirs(self, 
                   dask_workdir: str = 'dask_workdir', 
                   output_path: str = 'output_path', 
                   stats_path: str = 'stats_path'):
        if not os.path.isdir(self.local_base_dir):
            os.mkdir(self.local_base_dir)
        for dir_path in (dask_workdir, output_path, stats_path):
            if not os.path.isdir(self.local_base_dir/dir_path):
                os.mkdir(self.local_base_dir / dir_path)
            else:
                warnings.warn(f'Path {self.local_base_dir/dir_path} \
                                    already exists!')