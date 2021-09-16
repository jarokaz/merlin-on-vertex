# Copyright 2014 Google LLC
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


'''Wrapper around `gcloud alpha storage` tool to help perform download
or upload from/to GCS.
'''


from typing import Dict, List, Union
import subprocess


class GcsCopyUtils:
    '''Helper class to perform download/upload from/to GCS
    using `gcloud alpha storage` command line interface.

    Parameters
    ----------
        gcs_source: Union[str,List[str]]
            Path or a list of paths to be DOWNLOADED from GCS.
            The path can be a single file (one path), a list with multiple 
            files (multiple paths), one folder (one path), multiple folders 
            (multiple paths) or a combination of files and folders from GCS.
            Examples:
                (one file): 'gs://my_bucket/file.parquet'
                (multiple files): ['gs://my_bucket/file1.parquet',
                                   'gs://my_bucket/file1.parquet']
                (one folder): 'gs://my_bucket/subfolder'
                (multiple folders): ['gs://my_bucket/subfolder1',
                                     'gs://my_bucket/subfolder2']
                (combination): ['gs://my_bucket/file1.parquet',
                                'gs://my_bucket/subfolder2']
                There is an attribute specifically to indicate recursion 
                during download, but if a flag specified in the path, the flag 
                will be ignored. Examples:
                (all files, 1 level): gs://my_bucket/subfolder/*
                (all files, all levels): gs://my_bucket/subfolder/**
        local_download_path: str
            Single path in the local filesystem where objects will be 
            downloaded to. This must be a folder and the runtime must have 
            write access to it. Example: '/home/user/data'.
        local_upload_path: str
            Single path in the local filesystems which will be recursively 
            uploaded to GCS. The runtime must have read access to the path.
            Example: '/home/user/output'.
        gcs_dest: str
            Single path in google cloud storage where objects will be
            uploaded to. This must be a bucket or folder, and the runtime 
            must have write access to it. Examples:
                (bucket, root folder) 'gs://mybucket'
                (folder) 'gs://mybucket/myfolder'
        recursive: bool
            Variable used in the download operation. If set to True, will 
            search recursively for objects with a specific extension.
        extension: str = 'txt'
            Extension of the object to be downloaded. Examples: 'parquet', 
            'csv', 'txt', etc.
            This variable cannot be empty, otherwise, it will be treated as a
            folder in the path.
    '''

    def __init__(
        self, 
        gcs_source: Union[str,List[str]],
        local_download_path: str,
        local_upload_path: str,
        gcs_dest: str,
        recursive: bool = False,
        extension: str = 'parquet'
    ):
        if isinstance(gcs_source, list):
            self.gcs_source = gcs_source
        elif isinstance(gcs_source, str):
            self.gcs_source = [gcs_source]

        self.gcs_dest = gcs_dest
        self.local_download_path = local_download_path
        self.local_upload_path = local_upload_path
        self.recursive = recursive
        self.extension = extension

    def compose_gcloud_download_cmd(self) -> str:
        '''Composes a gcloud command for the download operation'''
        rec_symbol = '**' if self.recursive else '*'
        formated_paths = []

        for path in self.gcs_source:
            if path.endswith(f'.{self.extension}'):
                formated_paths.append(path)
            else:
                if path.endswith('/'):
                    formated_paths.append(
                        f'{path}{rec_symbol}.{self.extension}')
                elif (path.endswith('/*') or path.endswith('/**')):
                    formated_paths.append(f'{path}.{self.extension}')
                else:
                    formated_paths.append(
                        f'{path}/{rec_symbol}.{self.extension}')

        gcloud_cmd = ['gcloud', 'alpha', 'storage', 'cp', 
                        *formated_paths, self.local_download_path]

        return gcloud_cmd

    def compose_gcloud_upload_cmd(self) -> List[str]:
        '''Composes a gcloud command for the upload operation'''
        gcloud_cmd = ['gcloud', 'alpha', 'storage', 'cp', 
                            '-r', self.local_upload_path, self.gcs_dest]
        return gcloud_cmd

    def execute_gcloud_cmd(self, gcloud_cmd: List[str]) -> Dict[str,str]:
        '''Execute a gcloud alpha storage command and return 
        the stderr/stdout'''
        output = subprocess.run(gcloud_cmd, capture_output=True, text=True)
        return {'returncode': output.returncode,
                'stdout': output.stdout,
                'stderr': output.stderr}