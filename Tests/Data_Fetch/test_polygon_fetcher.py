import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.polygon_fetcher import PolygonDataFetcher

class TestPolygonDataFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = PolygonDataFetcher()
        self.sample_data = {
            "results": [
                {'t': 1625074800000, 'o': 150.0, 'h': 155.0, 'l': 148.0, 'c': 152.0, 'v': 1000000}
            ]
        }
        self.sample_real_time_data = {
            "results": {
                't': 1625074800000, 'price': 150.0, 'size': 1000
            }
        }

    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL", "2023-01-01", "2023-12-31")
        expected_url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31?apiKey=POLYGON_API_KEY"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [{
            'date': '2021-06-30',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    def test_construct_real_time_api_url(self):
        url = self.fetcher.construct_real_time_api_url("AAPL")
        expected_url = "https://api.polygon.io/v1/last/stocks/AAPL?apiKey=POLYGON_API_KEY"
        self.assertEqual(url, expected_url)

    def test_extract_real_time_results(self):
        results = self.fetcher.extract_real_time_results(self.sample_real_time_data)
        expected_results = [{
            'timestamp': '2021-06-30 00:00:00',
            'price': 150.0,
            'size': 1000
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'date')

    @patch('requests.get')
    def test_fetch_data_for_symbol_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
