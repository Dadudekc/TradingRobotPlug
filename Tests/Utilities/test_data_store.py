# C:\TheTradingRobotPlug\Tests\Utilities\test_data_store.py

import unittest
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import logging

# Ensure the project root is added to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Adjusted to correctly find the project root
sys.path.append(str(project_root))

from Scripts.Utilities.data_store import DataStore

class TestDataStore(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_csv_dir = 'test_data/csv'
        self.test_db_path = 'test_data/test_trading_data.db'
        self.test_data_store_path = 'test_data/data_store.pkl'
        self.test_csv_file = 'test_data.csv'
        self.test_table_name = 'test_table'
        self.test_logs_dir = 'logs'
        
        # Create necessary directories
        os.makedirs(self.test_csv_dir, exist_ok=True)
        os.makedirs(self.test_logs_dir, exist_ok=True)

        self.data_store = DataStore(csv_dir=self.test_csv_dir, db_path=self.test_db_path)

        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })

    def tearDown(self):
        """Clean up test environment after each test."""
        # Properly close the logger before removing the logs directory
        logging.shutdown()
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
        if os.path.exists(self.test_logs_dir):
            shutil.rmtree(self.test_logs_dir)

    def test_add_data(self):
        self.data_store.add_data('AAPL', {'price': 150})
        self.assertIn('AAPL', self.data_store.data)
        self.assertEqual(self.data_store.data['AAPL'], {'price': 150})

    def test_get_data(self):
        self.data_store.add_data('AAPL', {'price': 150})
        data = self.data_store.get_data('AAPL')
        self.assertEqual(data, {'price': 150})
        self.assertIsNone(self.data_store.get_data('MSFT'))

    def test_save_and_load_store(self):
        self.data_store.add_data('AAPL', {'price': 150})
        self.data_store.save_store(self.test_data_store_path)
        new_data_store = DataStore()
        new_data_store.load_store(self.test_data_store_path)
        self.assertEqual(new_data_store.data, {'AAPL': {'price': 150}})

    def test_save_to_csv(self):
        self.data_store.save_to_csv(self.test_df, self.test_csv_file, overwrite=True)
        file_path = os.path.join(self.test_csv_dir, self.test_csv_file)
        self.assertTrue(os.path.exists(file_path))
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, self.test_df)

    def test_fetch_from_csv(self):
        file_path = os.path.join(self.test_csv_dir, self.test_csv_file)
        self.test_df.to_csv(file_path, index=False)
        fetched_df = self.data_store.fetch_from_csv(self.test_csv_file)
        pd.testing.assert_frame_equal(fetched_df, self.test_df)

    def test_save_to_sql(self):
        self.data_store.save_to_sql(self.test_df, self.test_table_name, if_exists='replace')
        fetched_df = self.data_store.fetch_from_sql(self.test_table_name)
        pd.testing.assert_frame_equal(fetched_df, self.test_df)

    def test_list_csv_files(self):
        file_path = os.path.join(self.test_csv_dir, self.test_csv_file)
        self.test_df.to_csv(file_path, index=False)
        csv_files = self.data_store.list_csv_files()
        self.assertIn(self.test_csv_file, csv_files)

if __name__ == '__main__':
    unittest.main()
