# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\real_time_fetcher.py

import os
import sys
import requests
import pandas as pd
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import logging

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.base_fetcher import DataFetcher
from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.DataLakeHandler import DataLakeHandler

# Load environment variables from .env file
load_dotenv(dotenv_path="C:/TheTradingRobotPlug/.env")

# Set up logging
logging.basicConfig(level=logging.INFO, filename='real_time_data_fetcher.log', 
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeDataFetcher(DataFetcher):
    ALPHA_BASE_URL = "https://www.alphavantage.co/query"
    POLYGON_BASE_URL = "https://api.polygon.io/v1"

    def __init__(self, alpha_api_key: str, polygon_api_key: str):
        self.alpha_api_key = alpha_api_key
        self.polygon_api_key = polygon_api_key
        super().__init__('ALPHAVANTAGE_API_KEY', self.ALPHA_BASE_URL, 
                         'C:/TheTradingRobotPlug/data/real_time', 
                         'C:/TheTradingRobotPlug/data/processed_real_time', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/real_time.log', 
                         'AlphaVantageRealTime', None)

    def construct_alpha_api_url(self, symbol: str) -> str:
        return (
            f"{self.ALPHA_BASE_URL}?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}"
        )

    def construct_polygon_api_url(self, symbol: str) -> str:
        return (
            f"{self.POLYGON_BASE_URL}/last/stocks/{symbol}?apiKey={self.polygon_api_key}"
        )

    def extract_alpha_results(self, data: dict) -> list:
        if "Time Series (1min)" in data:
            return [
                {"timestamp": timestamp, **values}
                for timestamp, values in data["Time Series (1min)"].items()
            ]
        else:
            logger.error("Unexpected data format or error in Alpha Vantage response: %s", data)
            raise ValueError("Unexpected data format or error in response")

    def extract_polygon_results(self, data: dict) -> list:
        if "results" in data:
            return [
                {
                    "timestamp": datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": result['o'],
                    "high": result['h'],
                    "low": result['l'],
                    "close": result['c'],
                    "volume": result['v']
                }
                for result in data["results"]
            ]
        else:
            logger.error("Unexpected data format or error in Polygon response: %s", data)
            raise ValueError("Unexpected data format or error in response")

    def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        try:
            # Try fetching data from Alpha Vantage
            url = self.construct_alpha_api_url(symbol)
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info("Alpha Vantage API response data: %s", data)

            # Check for rate limit message
            if 'Information' in data and 'rate limit' in data['Information'].lower():
                raise RuntimeError("Alpha Vantage API rate limit has been reached. Switching to Polygon.")

            results = self.extract_alpha_results(data)
        except (requests.exceptions.HTTPError, ValueError, RuntimeError) as e:
            logger.warning("Alpha Vantage fetch failed: %s", e)
            # Fallback to Polygon API
            try:
                url = self.construct_polygon_api_url(symbol)
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                logger.info("Polygon API response data: %s", data)

                results = self.extract_polygon_results(data)
            except requests.exceptions.HTTPError as e:
                logger.error("Polygon API request failed: %s", e)
                if response.status_code == 403:
                    raise RuntimeError("Polygon API access forbidden: Check your API key and permissions.")
                else:
                    raise RuntimeError(f"Polygon API request failed: {e}")

        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        return df

# Example usage
if __name__ == "__main__":
    alpha_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    polygon_api_key = os.getenv('POLYGON_API_KEY')

    if not alpha_api_key:
        logger.error("Alpha Vantage API key is not set.")
    if not polygon_api_key:
        logger.error("Polygon API key is not set.")

    fetcher = RealTimeDataFetcher(alpha_api_key, polygon_api_key)
    try:
        df = fetcher.fetch_real_time_data("AAPL")
        print(df)
    except RuntimeError as e:
        logger.error(e)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
