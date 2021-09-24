from dataclasses import dataclass

@dataclass
class PathMetadata:
    is_directory: bool = False
    num_files: int = 0
    protocol: str = ''
    is_valid_path: bool = False
    is_path_exists: bool = False