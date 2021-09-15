import sys
sys.path.insert(1, '/home/renatoleite/workspace/merlin-on-vertex/src/io/')
from gcs_copy_utils import GcsCopyUtils

import unittest

class TestGcsCopyUtils(unittest.TestCase):
    def test_gcloud_cmd_download_single_file(self):
        gcs_source = 'gs://my_bucket/file.parquet'
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', 
                    'gs://my_bucket/file.parquet', '/data']
        
        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)
        
        self.assertEqual(gcs_util.compose_gcloud_download_cmd(), output)

    def test_gcloud_cmd_download_multiple_files(self):
        gcs_source = ['gs://my_bucket/file1.parquet',
                      'gs://my_bucket/file2.parquet']
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', 
                    'gs://my_bucket/file1.parquet',
                    'gs://my_bucket/file2.parquet', '/data']

        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)

        self.assertEqual(gcs_util.compose_gcloud_download_cmd(), output)

    def test_gcloud_cmd_download_single_folder(self):
        gcs_source = 'gs://my_bucket/folder'
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', 
                  'gs://my_bucket/folder/**.parquet', '/data']

        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)

        self.assertEqual(gcs_util.compose_gcloud_download_cmd(), output)

    def test_gcloud_cmd_download_multiple_folders(self):
        gcs_source = ['gs://my_bucket/folder1',
                      'gs://my_bucket/folder2']
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', 
                  'gs://my_bucket/folder1/**.parquet',
                  'gs://my_bucket/folder2/**.parquet','/data']

        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)

        self.assertEqual(gcs_util.compose_gcloud_download_cmd(), output)

    def test_gcloud_cmd_download_combination(self):
        gcs_source = ['gs://my_bucket/folder1',
                      'gs://my_bucket/file.parquet']
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', 
                  'gs://my_bucket/folder1/**.parquet',
                  'gs://my_bucket/file.parquet','/data']

        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)

        self.assertEqual(gcs_util.compose_gcloud_download_cmd(), output)

    def test_gcloud_cmd_download_upload(self):
        gcs_source = 'gs://my_bucket/folder1'
        local_download_path = '/data'
        local_upload_path = '/output'
        gcs_dest = 'gs://my_bucket/output'
        recursive = 'True'
        extension = 'parquet'
        
        output = ['gcloud', 'alpha', 'storage', 'cp', '-r',
                  '/output', 'gs://my_bucket/output']

        gcs_util = GcsCopyUtils(
                        gcs_source, local_download_path,
                        local_upload_path, gcs_dest,
                        recursive, extension)

        self.assertEqual(gcs_util.compose_gcloud_upload_cmd(), output)


if __name__ == '__main__':
    unittest.main()