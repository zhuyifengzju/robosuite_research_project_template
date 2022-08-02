import boto3

from pathlib import Path
import os

class S3Interface():
    def __init__(self):
        self._s3 = boto3.resource('s3')

    @staticmethod
    def exclude_path(path, exclude_list=[".git", "#", "~", "__pycache__"]):
        for exclude_str in exclude_list:
            if exclude_str in str(path):
                return True
        return False

    def list(self, target_folder_name, remote_folder_name="yifengz/bc", bucket_name="tri-gb"):

        bucket = self._s3.Bucket(bucket_name)
        remote_folder_path = Path(remote_folder_name)
        target_folder_path = Path(target_folder_name)
        path_list = []
        for obj in bucket.objects.filter(Prefix=str(remote_folder_path / target_folder_path) + "/"):

            path_name = str(obj.key.replace(remote_folder_name, "."))

            for path in list(Path(path_name).parents):
                if "run" in str(path) and "run" not in os.path.dirname(str(path)):
                    target_path = str(path)
                    if target_path not in path_list:
                        path_list.append(target_path)
        print(path_list)
        return path_list
    
    def upload_file(self, local_file_name, remote_folder_name="yifengz/bc", bucket_name="tri-gb"):
        remote_folder_path = Path(remote_folder_name)
        bucket = self._s3.Bucket(bucket_name)
        local_file = Path(local_file_name)
        bucket.upload_file(str(local_file), str(remote_folder_path / local_file))
    
    def upload_folder(self, local_folder_name, remote_folder_name="yifengz/bc", bucket_name="tri-gb"):
        remote_folder_path = Path(remote_folder_name)
        bucket = self._s3.Bucket(bucket_name)
        for path in Path(local_folder_name).glob('**/*'):
            if self.exclude_path(path):
                continue
            print(str(path))
            bucket.upload_file(str(path), str(remote_folder_path / path))

    def download_file(self, local_file_name, remote_folder_name="yifengz/bc", bucket_name="tri-gb"):
        bucket = self._s3.Bucket(bucket_name)
        remote_folder_path = Path(remote_folder_name)
        local_file_path = Path(local_file_name)
        # target_path = str(obj.key.replace(remote_folder_name, "./"))

        print("-----")
        print(str(remote_folder_path / local_file_path))
        print(local_file_name)
        if not os.path.exists(os.path.dirname(local_file_name)):
            os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
        bucket.download_file(str(remote_folder_path / local_file_path), local_file_name)
        
        # for obj in bucket.objects.filter(Prefix=str(remote_folder_path / local_file_path) + "/"):
        #     target_path = str(obj.key.replace(remote_folder_name, "./"))
        #     if not os.path.exists(os.path.dirname(target_path)):
        #         os.makedirs(os.path.dirname(target_path), exist_ok=True)
        #     if obj.key[-1] == '/':
        #         continue
        #     bucket.download_file(obj.key, target_path)
            
    def download_folder(self, local_folder_name, remote_folder_name="yifengz/bc", bucket_name="tri-gb"):
        bucket = self._s3.Bucket(bucket_name)

        remote_folder_path = Path(remote_folder_name)
        local_folder_path = Path(local_folder_name)
        for obj in bucket.objects.filter(Prefix=str(remote_folder_path / local_folder_path) + "/"):
            target_path = str(obj.key.replace(remote_folder_name, "./"))
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target_path)
