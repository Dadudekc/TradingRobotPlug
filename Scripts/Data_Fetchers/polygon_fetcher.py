import os
import sys
import pandas as pd
import aiohttp
from datetime import datetime
from typing import Optional, List, Dict, Any
from aiohttp import ClientSession, ClientTimeout
import asyncio
import logging
from pathlib import Path

# Ensure the project root is in the Python path for module imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from Scripts.Utilities.DataLakeHandler import DataLakeHandler
from Scripts.Data_Fetchers.base_fetcher import DataFetcher

class PolygonDataFetcher(DataFetcher):
    """
    Class for fetching historical and real-time data from the Polygon API.

    Attributes:
        data_lake_handler (Optional[DataLakeHandler]): Handler for storing data in a data lake.
    """

    def __init__(self, data_lake_handler: Optional[DataLakeHandler] = None):
        """
        Initializes the PolygonDataFetcher with the given parameters.
        
        Args:
            data_lake_handler (Optional[DataLakeHandler]): Handler for storing data in a data lake.
        """
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker',
                         'C:/TheTradingRobotPlug/data/polygon',
                         'C:/TheTradingRobotPlug/data/processed_polygon',
                         'C:/TheTradingRobotPlug/data/trading_data.db',
                         'C:/TheTradingRobotPlug/logs/polygon.log',
                         'Polygon', data_lake_handler)
        self.utils = self._initialize_utils()  # Initialize utils attribute

    def _initialize_utils(self):
        """
        Initializes utility attributes.
        
        Returns:
            None
        """
        # Placeholder for actual initialization of utility attributes
        return None

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        """
        Constructs the API URL for fetching data from Polygon.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date for fetching data.
            end_date (str): The end date for fetching data.

        Returns:
            str: The constructed API URL.
        """
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    async def fetch_data(self, url: str, session: ClientSession, retries: int = 3) -> Dict[str, Any]:
        """
        Fetches data from the provided URL with retries on failure.

        Args:
            url (str): The API URL to fetch data from.
            session (ClientSession): The aiohttp client session.
            retries (int): The number of retries on failure. Defaults to 3.

        Returns:
            Dict[str, Any]: The fetched data as a dictionary.
        """
        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data
            except aiohttp.ClientResponseError as e:
                if attempt < retries - 1 and e.status in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.utils.logger.error(f"Error fetching data from {url}: {e}")
                    raise
            except Exception as e:
                self.utils.logger.error(f"Unexpected error: {e}")
                raise

    def extract_results(self, data: dict, time_series_key: str = 'results') -> list:
        """
        Extracts results from the fetched data.

        Args:
            data (dict): The fetched data dictionary.
            time_series_key (str): The key for the time series data in the dictionary. Defaults to 'results'.

        Returns:
            list: A list of dictionaries containing the extracted results.
        """
        results = data.get(time_series_key, [])
        return [
            {
                'date': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

    async def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a given ticker symbol asynchronously.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date for fetching data.
            end_date (str): The end date for fetching data.

        Returns:
            Optional[pd.DataFrame]: The fetched data as a pandas DataFrame, or None if no data was fetched.
        """
        url = self.construct_api_url(symbol, start_date, end_date)
        timeout = ClientTimeout(total=60)
        
        try:
            if self.utils:
                self.utils.logger.debug(f"{self.source}: Request URL: {url}")
            async with ClientSession(timeout=timeout) as session:
                data = await self.fetch_data(url, session)
                if self.utils:
                    self.utils.logger.debug(f"{self.source}: Raw data for {symbol}: {data}")
                results = self.extract_results(data)
                
                if results:
                    df = pd.DataFrame(results)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()  # Ensure the index is sorted
                    df['symbol'] = symbol
                    filtered_df = df.loc[start_date:end_date]
                    if self.utils:
                        self.utils.logger.debug(f"{self.source}: Fetched data for {symbol}: {filtered_df}")
                    return filtered_df
                else:
                    if self.utils:
                        self.utils.logger.warning(f"{self.source}: Fetched data for {symbol} is not in the expected format. Data: {data}")
                    return None
        except Exception as e:
            self.utils.logger.error(f"Unexpected error for symbol {symbol}: {e}")
            return None

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetches real-time data for a given ticker symbol asynchronously.

        Args:
            symbol (str): The stock symbol to fetch real-time data for.

        Returns:
            pd.DataFrame: The fetched real-time data as a pandas DataFrame.
        """
        url = f"{self.base_url}/{symbol}/range/1/minute/2023-01-01/2023-12-31?apiKey={self.api_key}"
        timeout = ClientTimeout(total=60)
        
        try:
            if self.utils:
                self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
            async with ClientSession(timeout=timeout) as session:
                data = await self.fetch_data(url, session)
                results = self.extract_results(data)
                
                if results:
                    df = pd.DataFrame(results)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df['symbol'] = symbol
                    if self.utils:
                        self.utils.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                    return df
                else:
                    if self.utils:
                        self.utils.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format. Data: {data}")
                    return pd.DataFrame()
        except Exception as e:
            self.utils.logger.error(f"Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_data_for_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetches historical data for multiple ticker symbols asynchronously.

        Args:
            symbols (List[str]): The list of stock symbols to fetch data for.
            start_date (str): The start date for fetching data.
            end_date (str): The end date for fetching data.

        Returns:
            Dict[str, Optional[pd.DataFrame]]: A dictionary mapping symbols to their fetched data as pandas DataFrames.
        """
        timeout = ClientTimeout(total=60)
        async with ClientSession(timeout=timeout) as session:
            tasks = [self.fetch_data_for_symbol(symbol, start_date, end_date) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {symbol: result for symbol, result in zip(symbols, results)}

# Initialize logger for utility purposes
logging.basicConfig(level=logging.DEBUG)

# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\polygon_fetcher.py

import pandas as pd
import aiohttp
from datetime import datetime
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

class PolygonDataFetcher:
    """
    A class to fetch data from Polygon API.

    Attributes:
        api_key (str): The API key for Polygon.
        base_url (str): The base URL for Polygon API.
        logger (logging.Logger): Logger for logging messages.
    """

    def __init__(self):
        """
        Initializes the PolygonDataFetcher with API key and base URL.
        """
        self.api_key = "YOUR_POLYGON_API_KEY"
        self.base_url = "https://api.polygon.io/v2/aggs/ticker/"
        self.logger = DataFetchUtils("C:/TheTradingRobotPlug/logs/polygon_fetcher.log").logger

    async def fetch(self, symbol, start_date, end_date):
        """
        Fetches data for a given symbol from Polygon API.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date for fetching data.
            end_date (str): The end date for fetching data.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.
        """
        url = f"{self.base_url}{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "apiKey": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return self.extract_data(data)

    def extract_data(self, data):
        """
        Extracts data from the JSON response.

        Args:
            data (dict): The JSON response from the API.

        Returns:
            pd.DataFrame: The extracted data as a pandas DataFrame.
        """
        try:
            results = data.get("results", [])
            if not results:
                self.logger.warning(f"No data found for key: results")
                return pd.DataFrame()

            records = [
                {
                    "date": datetime.utcfromtimestamp(item["t"] / 1000).strftime('%Y-%m-%d'),
                    "open": item["o"],
                    "high": item["h"],
                    "low": item["l"],
                    "close": item["c"],
                    "volume": item["v"],
                    "symbol": data["ticker"]
                }
                for item in results
            ]
            df = pd.DataFrame(records)
            self.logger.info(f"Extracted {len(records)} records for symbol {data['ticker']}")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return pd.DataFrame()
