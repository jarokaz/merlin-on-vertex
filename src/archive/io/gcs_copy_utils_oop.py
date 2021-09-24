'''A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
'''
from local_path import LocalPath
from gcs_path import GcsPath

from typing import Dict, List, Union
import subprocess


class GcsCopyUtils:
    '''The Vehicle object contains a lot of vehicles

    Parameters
    ----------
        says_str : str
            a formatted string to print out what the animal says
        name : str
            the name of the animal
        sound : str
            the sound that the animal makes
        num_legs : int
            the number of legs the animal has (default 4)
    '''

    def __init__(
        self, 
        gcs_source: List[str],
        local_destination: str,
        gcs_destination: str,
        recursive: bool,
        extension: str,
        download_incomplete: bool
    ):
        self.download_incomplete = download_incomplete
        self.gcs_paths_source = self._generate_path_list(
                                            path_list=gcs_source,
                                            recursive=recursive, 
                                            extension=extension,
                                            location='gs',
                                            is_source=True)

        # TODO: "recursive" presents potential bug, verify.
        self.local_path_dest = self._generate_path_list(
                                            path_list=local_destination,
                                            recursive=recursive, 
                                            extension=extension,
                                            location='file',
                                            is_source=False)

        self.gcs_path_dest = self._generate_path_list(
                                            path_list=gcs_destination,
                                            recursive=recursive, 
                                            extension=extension,
                                            location='gs',
                                            is_source=False)

    def _generate_path_list(self, 
                            path_list: List[str], 
                            recursive: bool,
                            extension: str,
                            location: str, 
                            is_source: bool
                            ) -> List[Union[GcsPath,LocalPath]]:
        if location == 'gs':
            path_list = [GcsPath(path_name=path, 
                                 extension=extension,
                                 recursive=recursive,
                                 is_source=is_source) 
                                 for path in path_list]
        elif location == 'file':
            path_list = [LocalPath(path_name=path, 
                                 extension=extension,
                                 recursive=recursive,
                                 is_source=is_source) 
                                 for path in path_list]
        return path_list


    def generate_valid_path_list(self, 
        path_list: List[Union[GcsPath, LocalPath]]
                       ) -> List[Union[GcsPath, LocalPath]]:
        return [path for path in path_list 
                        if path.path_metadata.is_valid_path]


    def compose_gcloud_download_cmd(gcs_paths: List[GcsPath],
                                    local_destination: List[LocalPath], 
                                    extension: str = 'parquet', 
                                    recursive: bool = False) -> str:
        '''
        Valid paths:
            gs://my_bucket/file.parquet # file
            gs://my_bucket/subfolder or gs://my_bucket/subfolder/ # path
            gs://my_bucket/subfolder/* # all files, 1 level
            gs://my_bucket/subfolder/** # all files, all levels
        '''
        rec_symbol = '**' if recursive else '*'
        formated_paths = []

        for i in gcs_paths:
            if not i.path.path_metadata.is_directory:
                formated_paths.append(i.path.path_name)
            else:
                if i.path.path_name.endswith('/'):
                    formated_paths.append(
                        f'{i.path.path_name}{rec_symbol}.{extension}')
                elif (i.path.path_name.endswith('/*') 
                        or i.path.path_name.endswith('/**')):
                    formated_paths.append(f'{i.path.path_name}.{extension}')
                else:
                    formated_paths.append(
                        f'{i.path.path_name}/{rec_symbol}.{extension}')

        gcloud_cmd = ['gcloud', 'alpha', 'storage', 'cp', 
                        *formated_paths, local_destination[0].path.path_name]

        return gcloud_cmd


    def compose_gcloud_upload_cmd(local_path: str, 
                                  gcs_destination: str) -> List[str]:
        gcloud_cmd = ['gcloud', 'alpha', 'storage', 'cp', 
                        '-r', local_path, gcs_destination]
        return gcloud_cmd


    def execute_gcloud_cmd(gcloud_cmd: List) -> Dict[str,str]:
        output = subprocess.run(gcloud_cmd, capture_output=True, text=True)
        return {'returncode': output.returncode,
                'stdout': output.stdout,
                'stderr': output.stderr}


    def foo(self):
        '''Fetches rows from a Smalltable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by table_handle.  String keys will be UTF-8 encoded.

        Parameters
        ----------
        content_type: str
            If not None, set the content-type to this value
        content_encoding: str
            If not None, set the content-encoding.
            See https://cloud.google.com/storage/docs/transcoding
        kw_args: key-value pairs like field="value" or field=None
            value must be string to add or modify, or None to delete

        Returns
        -------
        Entire metadata after update (even if only path is passed)
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
        b'Zim': ('Irk', 'Invader'),
        b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).
        
        Raises
        ------
            IOError: An error occurred accessing the smalltable.
        '''
        pass
