import unittest
from unittest.mock import patch, Mock
import pandas as pd
import sys
import os

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetch.data_fetcher import AlphaVantageDataFetcher, NasdaqDataFetcher, PolygonDataFetcher

class TestDataFetcher(unittest.TestCase):

    @patch('Scripts.Data_Fetch.data_fetcher.requests.get')
    def test_alpha_vantage_fetch_data(self, mock_get):
        # Setup mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "100.0",
                    "2. high": "110.0",
                    "3. low": "90.0",
                    "4. close": "105.0",
                    "5. volume": "1000000"
                }
            }
        }
        mock_get.return_value = mock_response

        fetcher = AlphaVantageDataFetcher()
        data = fetcher.fetch_data(["AAPL"], "2023-01-01", "2023-01-01")

        print(f"Fetched data for AAPL: {data}")
        self.assertIn("AAPL", data)
        self.assertIsInstance(data["AAPL"], pd.DataFrame)
        self.assertEqual(len(data["AAPL"]), 1)

    @patch('Scripts.Data_Fetch.data_fetcher.requests.get')
    def test_nasdaq_fetch_data(self, mock_get):
        # Setup mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "t": 1609459200000,  # Timestamp for 2021-01-01
                    "o": 100.0,
                    "h": 110.0,
                    "l": 90.0,
                    "c": 105.0,
                    "v": 1000000
                }
            ]
        }
        mock_get.return_value = mock_response

        fetcher = NasdaqDataFetcher()
        data = fetcher.fetch_data(["AAPL"], "2021-01-01", "2021-01-01")

        print(f"Fetched data for AAPL: {data}")
        self.assertIn("AAPL", data)
        self.assertIsInstance(data["AAPL"], pd.DataFrame)
        self.assertEqual(len(data["AAPL"]), 1)

    @patch('Scripts.Data_Fetch.data_fetcher.requests.get')
    def test_polygon_fetch_data(self, mock_get):
        # Setup mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "t": 1609459200000,  # Timestamp for 2021-01-01
                    "o": 100.0,
                    "h": 110.0,
                    "l": 90.0,
                    "c": 105.0,
                    "v": 1000000
                }
            ]
        }
        mock_get.return_value = mock_response

        fetcher = PolygonDataFetcher()
        data = fetcher.fetch_data(["AAPL"], "2021-01-01", "2021-01-01")

        print(f"Fetched data for AAPL: {data}")
        self.assertIn("AAPL", data)
        self.assertIsInstance(data["AAPL"], pd.DataFrame)
        self.assertEqual(len(data["AAPL"]), 1)

if __name__ == '__main__':
    unittest.main()
