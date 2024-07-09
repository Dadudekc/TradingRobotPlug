# C:\TheTradingRobotPlug\Tests\Utilities\test_data_store.py

import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import pickle
from Scripts.Utilities.data_store import DataStore

class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.data_store = DataStore(csv_dir='test_data/csv', db_path='test_data/trading_data.db')
        self.test_data = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']}
        self.df = pd.DataFrame(self.test_data)
        self.test_pickle_file = 'test_data_store.pkl'
        self.test_csv_file = 'test_data.csv'
        self.test_table_name = 'test_table'
        
        # Ensure the test directories exist
        if not os.path.exists('test_data/csv'):
            os.makedirs('test_data/csv')

    def tearDown(self):
        if os.path.exists(self.test_pickle_file):
            os.remove(self.test_pickle_file)
        if os.path.exists(os.path.join('test_data/csv', self.test_csv_file)):
            os.remove(os.path.join('test_data/csv', self.test_csv_file))
        if os.path.exists(self.data_store.db_path):
            os.remove(self.data_store.db_path)
        if os.path.exists('test_data/csv'):
            os.rmdir('test_data/csv')

    def test_add_data(self):
        self.data_store.add_data('AAPL', self.test_data)
        self.assertIn('AAPL', self.data_store.data)
        self.assertEqual(self.data_store.data['AAPL'], self.test_data)

    def test_add_data_empty(self):
        with self.assertRaises(ValueError):
            self.data_store.add_data('AAPL', {})

    def test_add_data_wrong_type(self):
        with self.assertRaises(TypeError):
            self.data_store.add_data('AAPL', "not a dictionary")

    def test_get_data(self):
        self.data_store.data['AAPL'] = self.test_data
        data = self.data_store.get_data('AAPL')
        self.assertEqual(data, self.test_data)

    def test_save_store(self):
        self.data_store.data = {'AAPL': self.test_data}
        self.data_store.save_store(self.test_pickle_file)
        self.assertTrue(os.path.exists(self.test_pickle_file))

    def test_load_store(self):
        with open(self.test_pickle_file, 'wb') as file:
            pickle.dump({'AAPL': self.test_data}, file)
        self.data_store.load_store(self.test_pickle_file)
        self.assertIn('AAPL', self.data_store.data)
        self.assertEqual(self.data_store.data['AAPL'], self.test_data)

    @patch('Scripts.Utilities.data_fetch_utils.DataFetchUtils.save_data_to_csv')
    def test_save_to_csv(self, mock_save_to_csv):
        self.data_store.save_to_csv(self.df, self.test_csv_file, overwrite=True)
        mock_save_to_csv.assert_called_once()

    @patch('Scripts.Utilities.data_fetch_utils.DataFetchUtils.save_data_to_sql')
    def test_save_to_sql(self, mock_save_to_sql):
        self.data_store.save_to_sql(self.df, self.test_table_name)
        mock_save_to_sql.assert_called_once()

    @patch('pandas.read_csv')
    def test_fetch_from_csv(self, mock_read_csv):
        self.data_store.fetch_from_csv(self.test_csv_file)
        mock_read_csv.assert_called_once()

    @patch('Scripts.Utilities.data_fetch_utils.DataFetchUtils.fetch_data_from_sql')
    def test_fetch_from_sql(self, mock_fetch_data_from_sql):
        self.data_store.fetch_from_sql(self.test_table_name)
        mock_fetch_data_from_sql.assert_called_once()

    def test_list_csv_files(self):
        with open(os.path.join('test_data/csv', self.test_csv_file), 'w') as f:
            f.write("column1,column2\n1,a\n2,b\n3,c\n")
        csv_files = self.data_store.list_csv_files()
        self.assertIn(self.test_csv_file, csv_files)

if __name__ == '__main__':
    unittest.main()
