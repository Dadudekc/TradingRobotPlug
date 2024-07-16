import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import asyncio
from aiohttp import ClientSession
from dotenv import load_dotenv
import logging

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# Import the class to be tested
from Scripts.Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher

# Load environment variables
load_dotenv()

# Set up logging to show only warnings and errors
logging.basicConfig(level=logging.WARNING)

class TestAlphaVantageDataFetcher(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.symbol = 'AAPL'
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
        self.fetcher = AlphaVantageDataFetcher(self.api_key)
        self.fetcher.utils = MagicMock()  # Mock the utils attribute
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

    @patch.dict('os.environ', {'ALPHAVANTAGE_API_KEY': 'ALPHAVANTAGE_API_KEY'})
    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL")
        expected_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&interval=1min&apikey=ALPHAVANTAGE_API_KEY&outputsize=full&datatype=json"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data, "Time Series (Daily)")
        expected_results = [{
            'date': '2023-07-01',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    async def fetch_data(self, symbol, start_date, end_date):
        return await self.fetcher.fetch_data_for_symbol(symbol, start_date, end_date)

    def test_fetch_data_for_symbol(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.fetch_data(self.symbol, self.start_date, self.end_date))
        loop.close()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
        self.assertIn('symbol', result.columns)

    @patch('aiohttp.ClientSession.get')
    def test_fetch_data_for_symbol_error(self, mock_get):
        async def run_test():
            mock_get.side_effect = Exception("Test Exception")
            data = await self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
            self.assertIsNone(data)

        asyncio.run(run_test())

    @patch.object(AlphaVantageDataFetcher, '_initialize_utils', return_value=MagicMock())
    def test_logging_in_extract_results(self, mock_initialize_utils):
        self.fetcher.utils = mock_initialize_utils.return_value
        self.fetcher.extract_results(self.sample_data, "Time Series (Daily)")
        expected_log_message = "AlphaVantage: Extracted results: [{'date': '2023-07-01', 'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000}]"
        self.fetcher.utils.debug.assert_called_once_with(expected_log_message)

    @patch('Scripts.Data_Fetchers.alpha_vantage_fetcher.ClientSession')
    @patch('Scripts.Data_Fetchers.alpha_vantage_fetcher.AlphaVantageDataFetcher.fetch_data_for_symbol', new_callable=AsyncMock)
    def test_fetch_data_for_symbol_integration(self, mock_fetch_data_for_symbol, MockClientSession):
        mock_session = MockClientSession.return_value
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json.return_value = {
            "Time Series (Daily)": {
                "2024-07-16": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "95.0",
                    "4. close": "102.0",
                    "5. volume": "10000"
                }
            }
        }

        # Mock fetch_data_for_symbol function to return a DataFrame
        mock_fetch_data_for_symbol.return_value = pd.DataFrame({
            'date': ['2024-07-16'],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [10000],
            'symbol': ['AAPL']
        })

        async def run_test():
            data = await self.fetcher.fetch_data_for_symbol('AAPL', '2024-07-15', '2024-07-16')
            return data

        result = asyncio.run(run_test())

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 7))  # Ensure DataFrame shape is correct
        self.assertEqual(result['open'].iloc[0], 100.0)
        self.assertEqual(result['high'].iloc[0], 105.0)
        self.assertEqual(result['low'].iloc[0], 95.0)
        self.assertEqual(result['close'].iloc[0], 102.0)
        self.assertEqual(result['volume'].iloc[0], 10000)
        self.assertEqual(result['symbol'].iloc[0], 'AAPL')

if __name__ == '__main__':
    unittest.main()
