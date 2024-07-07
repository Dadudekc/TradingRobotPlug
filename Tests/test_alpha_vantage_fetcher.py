import os
import json
import unittest
from unittest.mock import patch, Mock
from datetime import datetime
from Scripts.DataFetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher

class TestAlphaVantageDataFetcher(unittest.TestCase):
    def setUp(self):
        os.environ['ALPHAVANTAGE_API_KEY'] = 'test_api_key'
        os.environ['CSV_DIR'] = '/tmp'
        os.environ['DB_PATH'] = '/tmp/test.db'
        os.environ['ALPHA_LOG_FILE'] = '/tmp/alpha.log'
        
        self.fetcher = AlphaVantageDataFetcher()

        self.mock_requests_get = patch('requests.get').start()
        self.addCleanup(patch.stopall)

    def test_fetch_data(self):
        symbol = "AAPL"
        start_date = "2022-01-01"
        end_date = "2022-12-31"

        response_data = {
            "Time Series (Daily)": {
                "2022-01-03": {
                    "1. open": "177.83",
                    "2. high": "182.88",
                    "3. low": "177.71",
                    "4. close": "182.01",
                    "5. volume": "104487900"
                }
            }
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
            "Time Series (1min)": {
                "2022-01-03 09:30:00": {
                    "1. open": "177.83",
                    "2. high": "177.88",
                    "3. low": "177.71",
                    "4. close": "177.01",
                    "5. volume": "487900"
                }
            }
        }

        self.mock_requests_get.return_value = Mock(status_code=200)
        self.mock_requests_get.return_value.json.return_value = response_data

        fetched_data = self.fetcher.fetch_real_time_data(symbol)

        self.assertIsNotNone(fetched_data)
        self.assertFalse(fetched_data.empty)

if __name__ == '__main__':
    unittest.main()
