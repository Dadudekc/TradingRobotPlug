# C:\TheTradingRobotPlug\Tests\Utilities\test_data_fetch_utils.py

import unittest
import os
import shutil
import sys
from pathlib import Path
import pandas as pd

# Ensure the project root is added to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Adjusted to correctly find the project root
sys.path.append(str(project_root))

from Scripts.Utilities.data_fetch_utils import DataFetchUtils

class TestDataFetchUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_csv_dir = 'test_data/csv'
        self.test_db_path = 'test_data/test_trading_data.db'
        self.test_csv_file = 'test_data.csv'
        self.test_table_name = 'test_table'
        self.data_utils = DataFetchUtils()
        
        # Create test data directory
        os.makedirs(self.test_csv_dir, exist_ok=True)

        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })

    def tearDown(self):
        """Clean up test environment after each test."""
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')

    def test_ensure_directory_exists(self):
        test_dir = 'test_data/new_dir'
        self.data_utils.ensure_directory_exists(test_dir)
        self.assertTrue(os.path.exists(test_dir))

    def test_save_data_to_csv(self):
        file_path = os.path.join(self.test_csv_dir, self.test_csv_file)
        self.data_utils.save_data_to_csv(self.test_df, file_path, overwrite=True)
        self.assertTrue(os.path.exists(file_path))
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, self.test_df)

    def test_save_data_to_sql(self):
        self.data_utils.save_data_to_sql(self.test_df, self.test_table_name, self.test_db_path, if_exists='replace')
        fetched_df = self.data_utils.fetch_data_from_sql(self.test_table_name, self.test_db_path)
        pd.testing.assert_frame_equal(fetched_df, self.test_df)

    def test_fetch_data_from_sql(self):
        self.data_utils.save_data_to_sql(self.test_df, self.test_table_name, self.test_db_path, if_exists='replace')
        fetched_df = self.data_utils.fetch_data_from_sql(self.test_table_name, self.test_db_path)
        pd.testing.assert_frame_equal(fetched_df, self.test_df)

if __name__ == '__main__':
    unittest.main()
