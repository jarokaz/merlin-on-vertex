from path import Path
from extension import Extension
from path_metadata import PathMetadata

from typing import Dict, Union, List

class GenericPath:
    """Base class for Path definition"""
    def __init__(self, 
                 path_name: str, 
                 extension: str, 
                 path_metadata: Dict[str,Union[bool,int,str]] = None):
        self.path = Path(path_name=path_name)
        self.extension = Extension(ext_name=extension)
        self.path_metadata = PathMetadata(**path_metadata)

    def _path_check(self):
        raise NotImplementedError("""Check if it is a valid path""")

    def _is_path_exists(self):
        raise NotImplementedError("""Check if path exists""")

    def _is_directory(self):
        raise NotImplementedError("""Is directory or file""")

    def _get_num_files(self):
        raise NotImplementedError("""Count number of files with extension in directory""")

    def _get_protocol(self):
        raise NotImplementedError("""Retrive protocol from file path""")