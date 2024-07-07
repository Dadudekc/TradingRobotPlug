import os
import json
import unittest
from unittest.mock import patch, Mock
from datetime import datetime
from Scripts.DataFetchers.polygon_fetcher import PolygonDataFetcher

class TestPolygonDataFetcher(unittest.TestCase):
    def setUp(self):
        os.environ['POLYGON_API_KEY'] = 'test_api_key'
        os.environ['CSV_DIR'] = '/tmp'
        os.environ['DB_PATH'] = '/tmp/test.db'
        os.environ['POLYGON_LOG_FILE'] = '/tmp/polygon.log'
        
        self.fetcher = PolygonDataFetcher()

        self.mock_requests_get = patch('requests.get').start()
        self.addCleanup(patch.stopall)

    def test_fetch_data(self):
        symbol = "AAPL"
        start_date = "2022-01-01"
        end_date = "2022-12-31"

        response_data = {
            "results": [
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
            "results": {
                "t": 1641205200000,
                "price": 177.83,
                "size": 487900
            }
        }

        self.mock_requests_get.return_value = Mock(status_code=200)
        self.mock_requests_get.return_value.json.return_value = response_data

        fetched_data = self.fetcher.fetch_real_time_data(symbol)

        self.assertIsNotNone(fetched_data)
        self.assertFalse(fetched_data.empty)

if __name__ == '__main__':
    unittest.main()
