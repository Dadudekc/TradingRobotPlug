# C:\TheTradingRobotPlug\Scripts\Utilities\DataLakeHandler.py

import boto3
from botocore.exceptions import NoCredentialsError
import logging

class DataLakeHandler:
    def __init__(self, bucket_name, region_name='us-west-2'):
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def upload_file(self, file_path, s3_path):
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_path)
            self.logger.info(f"File {file_path} uploaded to {s3_path} in bucket {self.bucket_name}.")
        except FileNotFoundError:
            self.logger.error(f"File {file_path} not found.")
        except NoCredentialsError:
            self.logger.error("Credentials not available.")

    def upload_data(self, data, s3_path):
        try:
            self.s3_client.put_object(Body=data, Bucket=self.bucket_name, Key=s3_path)
            self.logger.info(f"Data uploaded to {s3_path} in bucket {self.bucket_name}.")
        except NoCredentialsError:
            self.logger.error("Credentials not available.")
