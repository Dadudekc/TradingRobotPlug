# C:\TheTradingRobotPlug\Tests\Utilities\test_data_store.py

import unittest
import os
import tempfile
import shutil
import sys

# Add the path to the directory containing data_store.py
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(os.path.join(project_root, 'Scripts', 'Utilities'))

import Utilities.

class TestDataStore(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = DataStore(csv_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_add_data(self):
        self.store.add_data('AAPL', {'date': ['2023-01-01'], 'close': [150]})
        self.assertIn('AAPL', self.store.data)

    def test_get_data(self):
        self.store.add_data('AAPL', {'date': ['2023-01-01'], 'close': [150]})
        data = self.store.get_data('AAPL')
        self.assertEqual(data['close'], [150])

    @unittest.mock.patch('builtins.open', new_callable=unittest.mock.mock_open)
    @unittest.mock.patch('os.path.exists', return_value=True)
    def test_save_store(self, mock_exists, mock_file):
        self.store.add_data('AAPL', {'date': ['2023-01-01'], 'close': [150]})
        self.store.save_store('store.pkl')
        mock_file.assert_called_with('store.pkl', 'wb')

    @unittest.mock.patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_load_store(self, mock_file):
        self.store.load_store('store.pkl')
        mock_file.assert_called_with('store.pkl', 'rb')

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.store.add_data('AAPL', {})

    def test_invalid_data(self):
        with self.assertRaises(TypeError):
            self.store.add_data('AAPL', "Invalid Data")

if __name__ == '__main__':
    unittest.main()
