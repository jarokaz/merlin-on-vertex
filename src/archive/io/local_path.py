from generic_path import GenericPath
from fsspec import get_fs_token_paths

from typing import Dict, Union

class LocalPath(GenericPath):
    def __init__(self, 
                 path_name: str, 
                 extension: str, 
                 recursive: bool = False,
                 is_source: bool = True):
        # Fetch name protocol implementation for GCS
        self.fs_spec, _, _ = get_fs_token_paths(path_name)
        path_metadata = self._path_check(path_name, 
                                         extension, 
                                         recursive, 
                                         is_source)
        super().__init__(path_name=path_name, 
                         extension=extension, 
                         path_metadata=path_metadata)

    def _path_check(self, 
                    path_name: str, 
                    extension: str, 
                    recursive: bool,
                    is_source: bool) -> Dict[str,Union[bool,str,int]]:
        """Check if it is a valid path"""
        path_metadata = {}

        path_metadata['is_path_exists'] = self._is_path_exists(path_name)
        if path_metadata['is_path_exists']:
            path_metadata['protocol'] = self._get_protocol()
            if path_metadata['protocol'] == 'file':
                path_metadata['is_directory'] = self._is_directory(path_name)
                # Source must have at least one file with {extension}
                path_metadata['num_files'] = self._get_num_files(
                                                path_name,
                                                path_metadata['is_directory'],
                                                recursive,
                                                extension)
                if is_source:
                    if path_metadata['num_files'] > 0:
                        path_metadata['is_valid_path'] = True
                else:
                    if path_metadata['is_directory']:
                        path_metadata['is_valid_path'] = True

        return path_metadata

    def _is_path_exists(self, path_name: str) -> bool:
        """Check if path exists"""
        return self.fs_spec.exists(path_name)

    def _is_directory(self, path_name: str) -> bool:
        """Is directory or file"""
        return self.fs_spec.isdir(path_name)

    def _get_num_files(self, 
                       path_name: str, 
                       is_directory: bool, 
                       recursive: bool, 
                       extension: str) -> int:
        """Count number of files with extension in directory"""
        if not is_directory:
            if path_name.endswith(f'.{extension}'):
                return 1
            else:
                return 0
        else:
            if not path_name.endswith('/'):
                path_name = path_name + '/'
            if recursive:
                file_list = self.fs_spec.glob(f'{path_name}**.{extension}')
            else:
                file_list = self.fs_spec.glob(f'{path_name}*.{extension}')
            return len(file_list)

    def _get_protocol(self):
        """Retrive protocol from file path"""
        return self.fs_spec.protocol