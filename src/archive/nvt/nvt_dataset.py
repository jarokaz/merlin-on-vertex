# Use GCS Fuse or gs://


from typing import List, Dict, Union
from pathlib import Path
from glob import glob

import numpy as np
from dask.distributed import Client
import nvtabular as nvt


class DatasetNotCreated(Exception):
    pass


class NvtDataset:
    def __init__(self,
                data_path: str,
                engine: str,
                columns: List[str],
                cols_dtype: Dict[Union[np.dtype, str]],
                dataset_name: str,
                frac_size: float,
                shuffle: str = None,
                CATEGORICAL_COLUMNS: str = None,
                CONTINUOUS_COLUMNS: str = None,
                LABEL_COLUMNS: str = None):
        self.data_path = data_path
        self.engine = engine
        self.columns = columns
        self.cols_dtype = cols_dtype
        self.frac_size = frac_size
        self.shuffle = shuffle

    def create_nvt_dataset(self, 
                           client: Client,
                           separator: str = None):
        file_list = glob(str(Path(self.data_path) / 
                                f'*.{self.extension}'))

        self.dataset = nvt.Dataset(
            file_list,
            engine=self.extension,
            names=self.columns,
            part_mem_fraction=self.frac_size,
            sep=separator,
            dtypes=self.cols_dtypes,
            client=client,
        )
    
    def convert_to_parquet(self,
                           regen_nvt_dataset = False,
                           output_path: str = None,
                           client: Client = None,
                           preserve_files: bool = True):
        try:
            if output_path:
                self.dataset.to_parquet(
                    self.data_path,
                    preserve_files=preserve_files,
                    shuffle=self.shuffle
                )
            else:
                self.dataset.to_parquet(
                    output_path=output_path,
                    preserve_files=preserve_files,
                    shuffle=self.shuffle
                )
        except:
            raise DatasetNotCreated('Dataset not created')

        if regen_nvt_dataset:
            self.extension = 'parquet'
            if output_path:
                self.data_path = output_path
            self.create_nvt_dataset(client)