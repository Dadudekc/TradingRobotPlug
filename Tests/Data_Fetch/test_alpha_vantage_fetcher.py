# C:\TheTradingRobotPlug\Tests\Data_Fetch\test_alpha_vantage_fetcher.py

import os
import sys
import unittest
from unittest.mock import patch, AsyncMock
import pandas as pd
import asyncio

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher

class TestAlphaVantageDataFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = AlphaVantageDataFetcher()

    @patch('Scripts.Data_Fetchers.alpha_vantage_fetcher.AlphaVantageAPI.async_fetch_data')
    def test_fetch_data_success(self, mock_async_fetch_data):
        mock_response = {
            "Time Series (Daily)": {
                "2023-07-01": {
                    "1. open": "100.0",
                    "2. high": "110.0",
                    "3. low": "90.0",
                    "4. close": "105.0",
                    "5. volume": "1500000"
                }
            }
        }
        mock_async_fetch_data.return_value = mock_response

        async def run_test():
            df = await self.fetcher.fetch_data("AAPL")
            self.assertFalse(df.empty)
            self.assertEqual(df.loc['2023-07-01']['open'], 100.0)
            self.assertEqual(df.loc['2023-07-01']['high'], 110.0)

        asyncio.run(run_test())

    @patch('Scripts.Data_Fetchers.alpha_vantage_fetcher.AlphaVantageAPI.async_fetch_data')
    def test_fetch_data_no_results(self, mock_async_fetch_data):
        mock_async_fetch_data.return_value = None

        async def run_test():
            df = await self.fetcher.fetch_data("AAPL")
            self.assertTrue(df.empty)

        asyncio.run(run_test())

    def test_extract_results(self):
        data = {
            "Time Series (Daily)": {
                "2023-07-01": {
                    "1. open": "100.0",
                    "2. high": "110.0",
                    "3. low": "90.0",
                    "4. close": "105.0",
                    "5. volume": "1500000"
                }
            }
        }
        results = self.fetcher.extract_results(data)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['open'], 100.0)
        self.assertEqual(results[0]['high'], 110.0)

if __name__ == '__main__':
    unittest.main()
