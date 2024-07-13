# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\real_time_fetcher.py

import os
import sys
import requests
import pandas as pd
from typing import Optional
from datetime import datetime

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.data_fetcher import DataFetcher

class RealTimeDataFetcher(DataFetcher, RealTimeDataFetcherMixin):
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key):
        self.api_key = api_key
        self.utils = None  # Assuming there's a utils module with a logger

    def construct_real_time_api_url(self, symbol: str) -> str:
        return (
            f"{self.BASE_URL}?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}&interval=1min&apikey={self.api_key}"
        )

    def extract_real_time_results(self, data: dict) -> list:
        if "Time Series (1min)" in data:
            return [
                {"timestamp": timestamp, **values}
                for timestamp, values in data["Time Series (1min)"].items()
            ]
        else:
            raise ValueError("Unexpected data format or error in response")

# Example usage
# fetcher = RealTimeDataFetcher(api_key="your_api_key_here")
# df = fetcher.fetch_real_time_data("AAPL")
# print(df)
