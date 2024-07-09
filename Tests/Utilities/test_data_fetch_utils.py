# C:\TheTradingRobotPlug\Tests\Utilities\test_data_fetch_utils.py

import unittest
import os
import pandas as pd
import sqlite3
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

class TestDataFetchUtils(unittest.TestCase):

    def setUp(self):
        self.test_log_file = 'test_logs/data_fetch_utils.log'
        self.utils = DataFetchUtils(log_file=self.test_log_file)
        self.test_csv_file = 'test_data/test.csv'
        self.test_db_file = 'test_data/test.db'
        self.test_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        self.test_directory = 'test_data'

    def tearDown(self):
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        if os.path.exists(self.test_csv_file):
            os.remove(self.test_csv_file)
        if os.path.exists(self.test_db_file):
            os.remove(self.test_db_file)
        if os.path.exists(self.test_directory):
            os.rmdir(self.test_directory)
        log_dir = os.path.dirname(self.test_log_file)
        if os.path.exists(log_dir):
            os.rmdir(log_dir)

    def test_setup_logger(self):
        logger = self.utils.setup_logger("TestLogger", self.test_log_file)
        self.assertEqual(logger.name, "TestLogger")
        self.assertTrue(logger.hasHandlers())

    def test_ensure_directory_exists(self):
        self.utils.ensure_directory_exists(self.test_directory)
        self.assertTrue(os.path.exists(self.test_directory))

    def test_save_data_to_csv(self):
        self.utils.save_data_to_csv(self.test_data, self.test_csv_file, overwrite=True)
        self.assertTrue(os.path.exists(self.test_csv_file))
        df = pd.read_csv(self.test_csv_file)
        pd.testing.assert_frame_equal(df, self.test_data)

    def test_save_data_to_csv_exists_error(self):
        self.utils.save_data_to_csv(self.test_data, self.test_csv_file, overwrite=True)
        with self.assertRaises(FileExistsError):
            self.utils.save_data_to_csv(self.test_data, self.test_csv_file)

    def test_save_data_to_sql(self):
        self.utils.save_data_to_sql(self.test_data, 'test_table', self.test_db_file)
        conn = sqlite3.connect(self.test_db_file)
        df = pd.read_sql('SELECT * FROM test_table', conn)
        conn.close()
        pd.testing.assert_frame_equal(df, self.test_data)

    def test_fetch_data_from_sql(self):
        self.utils.save_data_to_sql(self.test_data, 'test_table', self.test_db_file)
        df = self.utils.fetch_data_from_sql('test_table', self.test_db_file)
        pd.testing.assert_frame_equal(df, self.test_data)

if __name__ == '__main__':
    unittest.main()
