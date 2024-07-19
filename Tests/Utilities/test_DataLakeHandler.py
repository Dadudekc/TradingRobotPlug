import unittest
from unittest.mock import patch, MagicMock
from moto import mock_s3
import boto3
import logging
import os

from Scripts.Utilities.DataLakeHandler import DataLakeHandler

class TestDataLakeHandler(unittest.TestCase):

    def setUp(self):
        self.bucket_name = "test-bucket"
        self.region_name = "us-west-2"

        # Start the mock S3 service
        self.mock_s3 = mock_s3()
        self.mock_s3.start()

        # Create a mock S3 client and bucket
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.s3_client.create_bucket(Bucket=self.bucket_name)

        self.data_lake_handler = DataLakeHandler(self.bucket_name, self.region_name)

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def tearDown(self):
        self.mock_s3.stop()

    @patch('boto3.client')
    def test_upload_file_success(self, mock_boto_client):
        # Mock the upload_file method
        mock_boto_client.return_value.upload_file = MagicMock()

        file_path = "test_file.txt"
        s3_path = "test_s3_path.txt"

        # Create a temporary test file
        with open(file_path, 'w') as f:
            f.write("This is a test file.")

        self.data_lake_handler.upload_file(file_path, s3_path)
        self.data_lake_handler.s3_client.upload_file.assert_called_with(file_path, self.bucket_name, s3_path)

        # Clean up the temporary test file
        os.remove(file_path)

    @patch('boto3.client')
    def test_upload_file_not_found(self, mock_boto_client):
        # Mock the upload_file method to raise FileNotFoundError
        mock_boto_client.return_value.upload_file.side_effect = FileNotFoundError

        file_path = "non_existent_file.txt"
        s3_path = "test_s3_path.txt"

        with self.assertLogs(self.data_lake_handler.logger, level='ERROR') as log:
            self.data_lake_handler.upload_file(file_path, s3_path)
            self.assertIn(f"File {file_path} not found.", log.output[0])

    @patch('boto3.client')
    def test_upload_file_no_credentials(self, mock_boto_client):
        # Mock the upload_file method to raise NoCredentialsError
        mock_boto_client.return_value.upload_file.side_effect = NoCredentialsError

        file_path = "test_file.txt"
        s3_path = "test_s3_path.txt"

        # Create a temporary test file
        with open(file_path, 'w') as f:
            f.write("This is a test file.")

        with self.assertLogs(self.data_lake_handler.logger, level='ERROR') as log:
            self.data_lake_handler.upload_file(file_path, s3_path)
            self.assertIn("Credentials not available.", log.output[0])

        # Clean up the temporary test file
        os.remove(file_path)

    @patch('boto3.client')
    def test_upload_data_success(self, mock_boto_client):
        # Mock the put_object method
        mock_boto_client.return_value.put_object = MagicMock()

        data = "This is a test data."
        s3_path = "test_s3_path.txt"

        self.data_lake_handler.upload_data(data, s3_path)
        self.data_lake_handler.s3_client.put_object.assert_called_with(Body=data, Bucket=self.bucket_name, Key=s3_path)

    @patch('boto3.client')
    def test_upload_data_no_credentials(self, mock_boto_client):
        # Mock the put_object method to raise NoCredentialsError
        mock_boto_client.return_value.put_object.side_effect = NoCredentialsError

        data = "This is a test data."
        s3_path = "test_s3_path.txt"

        with self.assertLogs(self.data_lake_handler.logger, level='ERROR') as log:
            self.data_lake_handler.upload_data(data, s3_path)
            self.assertIn("Credentials not available.", log.output[0])

if __name__ == '__main__':
    unittest.main()
