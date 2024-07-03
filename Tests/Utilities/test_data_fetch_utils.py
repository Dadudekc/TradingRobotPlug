import unittest
from Utilities import data_fetch_utils

class TestDataFetchUtils(unittest.TestCase):
    
    def test_fetch_data(self):
        result = data_fetch_utils.fetch_data('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_process_fetched_data(self):
        raw_data = {'date': ['2023-01-01'], 'close': [150]}
        processed_data = data_fetch_utils.process_fetched_data(raw_data)
        self.assertIn('processed', processed_data)

    def test_save_fetched_data(self):
        data = {'date': ['2023-01-01'], 'close': [150]}
        data_fetch_utils.save_fetched_data('AAPL', data)
        self.assertTrue(os.path.exists('data/AAPL.csv'))

if __name__ == '__main__':
    unittest.main()
