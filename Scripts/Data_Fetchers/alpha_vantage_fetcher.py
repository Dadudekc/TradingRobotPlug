# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\alpha_vantage_fetcher.py

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import aiohttp
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, ClientConnectionError, ContentTypeError

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Data_Fetchers.base_fetcher import DataFetcher
from Scripts.Utilities.DataLakeHandler import DataLakeHandler  # Add this import if `DataLakeHandler` is being used

class AlphaVantageDataFetcher(DataFetcher):
    def __init__(self, data_lake_handler: Optional[DataLakeHandler] = None):
        super().__init__('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/query', 
                         'C:/TheTradingRobotPlug/data/alpha_vantage', 
                         'C:/TheTradingRobotPlug/data/processed_alpha_vantage', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/alpha_vantage.log', 
                         'AlphaVantage', data_lake_handler)
        self.utils = DataFetchUtils("C:/TheTradingRobotPlug/logs/alpha_vantage_fetcher.log").logger

    def construct_api_url(self, symbol: str, function: str = "TIME_SERIES_DAILY", interval: str = "1min") -> str:
        return f"{self.base_url}?function={function}&symbol={symbol}&interval={interval}&apikey={self.api_key}&outputsize=full&datatype=json"

    async def fetch_data(self, url: str, session: ClientSession, retries: int = 3) -> Dict[str, Any]:
        for attempt in range(retries):
            try:
                self.utils.debug(f"Fetching data from URL: {url}, Attempt: {attempt+1}")
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self.utils.debug(f"Fetched data: {data}")  # Log fetched data
                    return data
            except aiohttp.ClientResponseError as e:
                self.utils.error(f"ClientResponseError: {e}, Status: {e.status}")
                if attempt < retries - 1 and e.status in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.utils.error(f"Error fetching data from {url}: {e}")
                    raise
            except Exception as e:
                self.utils.error(f"Unexpected error: {e}")
                raise

    def extract_results(self, data: dict, time_series_key: str) -> list:
        time_series = data.get(time_series_key, {})
        if not time_series:
            self.utils.debug(f"No data found for key: {time_series_key}")
        results = [
            {
                'date': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]
        self.utils.debug(f"AlphaVantage: Extracted results: {results}")
        return results

    async def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol)
        timeout = ClientTimeout(total=60)

        try:
            async with ClientSession(timeout=timeout) as session:
                data = await self.fetch_data(url, session)
                if not data:
                    self.utils.warning(f"{self.source}: No data fetched for {symbol}.")
                    return None

                results = self.extract_results(data, "Time Series (Daily)")

                if not results:
                    self.utils.warning(f"{self.source}: Fetched data for {symbol} is not in the expected format. Data: {data}")
                    return None

                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()  # Ensure the index is sorted
                df['symbol'] = symbol

                filtered_df = df.loc[start_date:end_date]
                self.utils.debug(f"{self.source}: Fetched data for {symbol}: {filtered_df}")

                return filtered_df

        except ClientConnectionError as e:
            self.utils.error(f"Connection error for symbol {symbol}: {e}")
        except ContentTypeError as e:
            self.utils.error(f"Unexpected content type for symbol {symbol}: {e}")
        except asyncio.TimeoutError:
            self.utils.error(f"Timeout error for symbol {symbol}")
        except Exception as e:
            self.utils.error(f"Unexpected error for symbol {symbol}: {e}")

        return None

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = self.construct_api_url(symbol, function="TIME_SERIES_INTRADAY", interval="1min")
        timeout = ClientTimeout(total=60)
        
        try:
            self.utils.debug(f"{self.source}: Real-time request URL: {url}")
            async with ClientSession(timeout=timeout) as session:
                data = await self.fetch_data(url, session)
                results = self.extract_results(data, "Time Series (1min)")
                
                if results:
                    df = pd.DataFrame(results)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df['symbol'] = symbol
                    self.utils.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                    return df
                else:
                    self.utils.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format. Data: {data}")
                    return pd.DataFrame()
        except Exception as e:
            self.utils.error(f"Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_data_for_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        timeout = ClientTimeout(total=60)
        async with ClientSession(timeout=timeout) as session:
            tasks = [self.fetch_data_for_symbol(symbol, start_date, end_date) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {symbol: result for symbol, result in zip(symbols, results)}

# Initialize logger for utility purposes
logging.basicConfig(level=logging.DEBUG)
