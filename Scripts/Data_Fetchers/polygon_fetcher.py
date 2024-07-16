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
    def __init__(self, data_lake_handler: Optional[DataLakeHandler] = None):
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker', 
                         'C:/TheTradingRobotPlug/data/polygon', 
                         'C:/TheTradingRobotPlug/data/processed_polygon', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/polygon.log', 
                         'Polygon', data_lake_handler)
        self.utils = self._initialize_utils()  # Initialize utils attribute

    def _initialize_utils(self):
        # Placeholder for actual initialization of utility attributes
        return None

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    async def fetch_data(self, url: str, session: ClientSession, retries: int = 3) -> Dict[str, Any]:
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
        timeout = ClientTimeout(total=60)
        async with ClientSession(timeout=timeout) as session:
            tasks = [self.fetch_data_for_symbol(symbol, start_date, end_date) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {symbol: result for symbol, result in zip(symbols, results)}

# Initialize logger for utility purposes
logging.basicConfig(level=logging.DEBUG)
