import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.real_time_fetcher import RealTimeDataFetcher

class TestRealTimeDataFetcher(RealTimeDataFetcher):
    def __init__(self):
        super().__init__()

    def construct_real_time_api_url(self, symbol: str) -> str:
        return f"https://api.example.com/real-time/{symbol}?apiKey=FAKE_API_KEY"

    def extract_real_time_results(self, data: dict) -> list:
        results = data.get('data', [])
        return [
            {
                'timestamp': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'price': result['price'],
                'size': result['size']
            }
            for result in results
        ]

class TestRealTimeDataFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = TestRealTimeDataFetcher()
        self.sample_real_time_data = {
            "data": [
                {'t': 1625074800000, 'price': 150.0, 'size': 100}
            ]
        }

    def test_construct_real_time_api_url(self):
        url = self.fetcher.construct_real_time_api_url("AAPL")
        expected_url = "https://api.example.com/real-time/AAPL?apiKey=FAKE_API_KEY"
        self.assertEqual(url, expected_url)

    def test_extract_real_time_results(self):
        results = self.fetcher.extract_real_time_results(self.sample_real_time_data)
        expected_results = [{
            'timestamp': '2021-06-30 00:00:00',
            'price': 150.0,
            'size': 100
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_real_time_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_real_time_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_real_time_data('AAPL')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'timestamp')

    @patch('requests.get')
    def test_fetch_real_time_data_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_real_time_data('AAPL')
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()

