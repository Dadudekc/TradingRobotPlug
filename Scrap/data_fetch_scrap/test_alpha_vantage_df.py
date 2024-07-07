# C:\TheTradingRobotPlug\Tests\Data_Fetch\test_alpha_vantage_df.py
# Tests\Data_Fetch\test_alpha_vantage_df.py
# To Run:
# 1st: cd C:\TheTradingRobotPlug\Tests
# 2nd: python -m unittest Data_Fetch.test_alpha_vantage_df

import unittest
from unittest.mock import patch, mock_open
import os
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Scripts/Data_Fetch'))
from alpha_vantage_df import AlphaVantageDataFetcher

class TestAlphaVantageDataFetcher(unittest.TestCase):

    @patch.dict(os.environ, {'ALPHAVANTAGE_API_KEY': 'mock_api_key'})
    @patch('alpha_vantage_df.requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        # Sample data to return from the mock API call
        mock_response = {
            "Time Series (Daily)": {
                "2023-01-01": {"1. open": "145.83", "2. high": "146.92", "3. low": "145.67", "4. close": "146.83", "5. volume": "53480044"}
            }
        }
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None

        fetcher = AlphaVantageDataFetcher()
        df = fetcher.fetch_data_for_symbol("AAPL", "2023-01-01", "2023-12-31")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(float(df.iloc[0]['open']), 145.83)
        self.assertEqual(float(df.iloc[0]['high']), 146.92)
        self.assertEqual(float(df.iloc[0]['low']), 145.67)
        self.assertEqual(float(df.iloc[0]['close']), 146.83)
        self.assertEqual(int(df.iloc[0]['volume']), 53480044)

    @patch.dict(os.environ, {'ALPHAVANTAGE_API_KEY': 'mock_api_key'})
    @patch('alpha_vantage_df.os.path.exists')
    @patch('alpha_vantage_df.pd.read_csv')
    @patch('alpha_vantage_df.AlphaVantageDataFetcher.fetch_data_for_symbol')
    @patch('alpha_vantage_df.pd.DataFrame.to_csv')
    def test_fetch_data(self, mock_to_csv, mock_fetch_data, mock_read_csv, mock_exists):
        mock_exists.side_effect = lambda x: False if "AAPL" in x else True

        mock_read_csv.return_value = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01']),
            'open': [145.83],
            'high': [146.92],
            'low': [145.67],
            'close': [146.83],
            'volume': [53480044]
        })

        mock_fetch_data.return_value = pd.DataFrame({
            'date': pd.to_datetime(['2023-02-01']),
            'open': [147.83],
            'high': [148.92],
            'low': [147.67],
            'close': [148.83],
            'volume': [63480044]
        })

        ticker_symbols = ["AAPL"]
        fetcher = AlphaVantageDataFetcher()
        data_frames = fetcher.fetch_data(ticker_symbols, "2023-01-01", "2023-12-31")

        self.assertIn("AAPL", data_frames)
        self.assertEqual(data_frames["AAPL"].iloc[0]['open'], 147.83)
        self.assertEqual(data_frames["AAPL"].iloc[0]['high'], 148.92)

        print(f"Debug: Data frames: {data_frames}")

        # Simulate saving to CSV
        for symbol, df in data_frames.items():
            fetcher.save_data_to_csv(df, symbol)

        # Check if to_csv was called
        self.assertTrue(mock_to_csv.called)
        print(f"Debug: mock_to_csv called: {mock_to_csv.called}")

if __name__ == "__main__":
    unittest.main()
