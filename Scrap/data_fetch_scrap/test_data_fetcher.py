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

from Scrap.data_fetch_scrap.data_fetcher import AlphaVantageDataFetcher, NasdaqDataFetcher, PolygonDataFetcher

class TestAlphaVantageDataFetcher(unittest.TestCase):

    def setUp(self):
        with patch.dict('os.environ', {'ALPHAVANTAGE_API_KEY': 'ALPHAVANTAGE_API_KEY'}):
            self.fetcher = AlphaVantageDataFetcher()
        self.sample_data = {
            "Time Series (Daily)": {
                "2023-07-01": {
                    "1. open": "150.00",
                    "2. high": "155.00",
                    "3. low": "148.00",
                    "4. close": "152.00",
                    "5. volume": "1000000"
                }
            }
        }

    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL", "2023-01-01", "2023-12-31")
        expected_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=ALPHAVANTAGE_API_KEY&outputsize=full&datatype=json"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [{
            'date': '2023-07-01',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        print(data)  # Debug statement
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'date')

    @patch('requests.get')
    def test_fetch_data_for_symbol_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

    @patch('Scripts.Utilities.data_store.DataStore.save_to_csv')
    @patch('Scripts.Utilities.data_store.DataStore.save_to_sql')
    def test_save_data(self, mock_save_to_sql, mock_save_to_csv):
        df = pd.DataFrame(self.sample_data['Time Series (Daily)']).T
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        self.fetcher.save_data(df, 'AAPL')
        mock_save_to_csv.assert_called_once()
        mock_save_to_sql.assert_called_once()

    @patch('Scripts.Utilities.data_store.DataStore.save_to_csv')
    @patch('Scripts.Utilities.data_store.DataStore.save_to_sql')
    def test_save_data_empty(self, mock_save_to_sql, mock_save_to_csv):
        df = pd.DataFrame()
        self.fetcher.save_data(df, 'AAPL')
        mock_save_to_csv.assert_not_called()
        mock_save_to_sql.assert_not_called()

class TestNasdaqDataFetcher(unittest.TestCase):

    def setUp(self):
        with patch.dict('os.environ', {'NASDAQ_API_KEY': 'NASDAQ_API_KEY'}):
            self.fetcher = NasdaqDataFetcher()
        self.sample_data = {
            "data": [
                {'t': 1672531200000, 'o': 150.0, 'h': 155.0, 'l': 148.0, 'c': 152.0, 'v': 1000000}
            ]
        }

    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL", "2023-01-01", "2023-12-31")
        expected_url = "https://dataondemand.nasdaq.com/api/v1/historical/AAPL?apiKey=NASDAQ_API_KEY&startDate=2023-01-01&endDate=2023-12-31"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [{
            'date': '2023-01-01',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        print(data)  # Debug statement
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'date')

    @patch('requests.get')
    def test_fetch_data_for_symbol_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

class TestPolygonDataFetcher(unittest.TestCase):

    def setUp(self):
        with patch.dict('os.environ', {'POLYGON_API_KEY': 'POLYGON_API_KEY'}):
            self.fetcher = PolygonDataFetcher()
        self.sample_data = {
            "results": [
                {'t': 1672531200000, 'o': 150.0, 'h': 155.0, 'l': 148.0, 'c': 152.0, 'v': 1000000}
            ]
        }

    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL", "2023-01-01", "2023-12-31")
        expected_url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31?apiKey=POLYGON_API_KEY"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [{
            'date': '2023-01-01',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        print(data)  # Debug statement
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
