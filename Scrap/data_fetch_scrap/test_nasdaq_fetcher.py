import os
import json
import unittest
from unittest.mock import patch, Mock
from datetime import datetime
from Scripts.DataFetchers.nasdaq_fetcher import NasdaqDataFetcher

class TestNasdaqDataFetcher(unittest.TestCase):
    def setUp(self):
        os.environ['NASDAQ_API_KEY'] = 'test_api_key'
        os.environ['CSV_DIR'] = '/tmp'
        os.environ['DB_PATH'] = '/tmp/test.db'
        os.environ['NASDAQ_LOG_FILE'] = '/tmp/nasdaq.log'
        
        self.fetcher = NasdaqDataFetcher()

        self.mock_requests_get = patch('requests.get').start()
        self.addCleanup(patch.stopall)

    def test_fetch_data(self):
        symbol = "AAPL"
        start_date = "2022-01-01"
        end_date = "2022-12-31"

        response_data = {
            "data": [
                {
                    "t": 1641205200000,
                    "o": 177.83,
                    "h": 182.88,
                    "l": 177.71,
                    "c": 182.01,
                    "v": 104487900
                }
            ]
        }

        self.mock_requests_get.return_value = Mock(status_code=200)
        self.mock_requests_get.return_value.json.return_value = response_data

        fetched_data = self.fetcher.fetch_data([symbol], start_date, end_date)

        self.assertIsNotNone(fetched_data)
        self.assertIn(symbol, fetched_data)
        self.assertFalse(fetched_data[symbol].empty)

    def test_fetch_real_time_data(self):
        symbol = "AAPL"
        response_data = {
            "data": [
                {
                    "t": 1641205200000,
                    "o": 177.83,
                    "h": 177.88,
                    "l": 177.71,
                    "c": 177.01,
                    "v": 487900
                }
            ]
        }

        self.mock_requests_get.return_value = Mock(status_code=200)
        self.mock_requests_get.return_value.json.return_value = response_data

        fetched_data = self.fetcher.fetch_real_time_data(symbol)

        self.assertIsNotNone(fetched_data)
        self.assertFalse(fetched_data.empty)

if __name__ == '__main__':
    unittest.main()
