# C:\TheTradingRobotPlug\Tests\Data_Fetch\test_polygon_io.py

# To Run:
# 1st: cd C:\TheTradingRobotPlug\Tests
# 2nd: python -m unittest Data_Fetch.test_polygon_io


import unittest
from unittest.mock import patch, mock_open
import os
import pandas as pd
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Scripts/Data_Fetch'))
from polygon_io import PolygonDataFetcher

class TestPolygonIO(unittest.TestCase):

    @patch('polygon_io.requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        # Sample data to return from the mock API call
        mock_response = {
            "results": [
                {"t": 1632182400000, "o": 145.83, "h": 146.92, "l": 145.67, "c": 146.83, "v": 53480044}
            ]
        }
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None

        fetcher = PolygonDataFetcher(api_key="mock_api_key", csv_dir="mock_csv_dir")
        df = fetcher.fetch_data_for_symbol("AAPL", "2023-01-01", "2023-12-31")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.iloc[0]['o'], 145.83)
        self.assertEqual(df.iloc[0]['h'], 146.92)
        self.assertEqual(df.iloc[0]['l'], 145.67)
        self.assertEqual(df.iloc[0]['c'], 146.83)
        self.assertEqual(df.iloc[0]['v'], 53480044)

    @patch('polygon_io.os.path.exists')
    @patch('polygon_io.pd.read_csv')
    @patch('polygon_io.PolygonDataFetcher.fetch_data_for_symbol')
    @patch('polygon_io.pd.DataFrame.to_csv')
    def test_fetch_data_from_polygon(self, mock_to_csv, mock_fetch_data, mock_read_csv, mock_exists):
        mock_exists.side_effect = lambda x: False if "AAPL" in x else True

        mock_read_csv.return_value = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01']),
            'o': [145.83],
            'h': [146.92],
            'l': [145.67],
            'c': [146.83],
            'v': [53480044]
        })

        mock_fetch_data.return_value = pd.DataFrame({
            'date': pd.to_datetime(['2023-02-01']),
            'o': [147.83],
            'h': [148.92],
            'l': [147.67],
            'c': [148.83],
            'v': [63480044]
        })

        ticker_symbols = ["AAPL", "MSFT"]
        fetcher = PolygonDataFetcher(api_key="mock_api_key", csv_dir="mock_csv_dir")
        data_frames = fetcher.fetch_data_from_polygon(ticker_symbols, "2023-01-01", "2023-12-31")

        self.assertIn("AAPL", data_frames)
        self.assertIn("MSFT", data_frames)
        self.assertEqual(data_frames["AAPL"].iloc[0]['o'], 147.83)
        self.assertEqual(data_frames["MSFT"].iloc[0]['o'], 145.83)

        mock_to_csv.assert_called()

if __name__ == "__main__":
    unittest.main()
