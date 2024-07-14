import os
import sys
import asyncio
import pandas as pd
import aiohttp
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

# Add project root to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from Scripts.Data_Fetchers.base_fetcher import DataFetcher
from Scripts.Utilities.data_store import DataStore

class PolygonDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('POLYGON_API_KEY', 
                         'https://api.polygon.io/v2/aggs/ticker', 
                         'C:/TheTradingRobotPlug/data/raw/polygon', 
                         'C:/TheTradingRobotPlug/data/processed/polygon', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/polygon.log', 
                         'Polygon')
        self.data_store = DataStore()

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    def extract_results(self, data: dict) -> list:
        results = data.get('results', [])
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

    async def async_fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        url = self.construct_api_url(symbol, start_date, end_date)
        try:
            self.logger.debug(f"{self.source}: Request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = self.extract_results(data)
                    if results:
                        df = pd.DataFrame(results)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df['symbol'] = symbol
                        self.logger.debug(f"{self.source}: Fetched data for {symbol}: {df}")
                        return df
                    else:
                        self.logger.warning(f"{self.source}: Data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"{self.source}: Error fetching data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={self.api_key}"

        try:
            self.logger.debug(f"{self.source}: Real-time request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = self.extract_real_time_results(data)

                    if results:
                        df = pd.DataFrame(results)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        df['symbol'] = symbol
                        self.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                        return df
                    else:
                        self.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"{self.source}: Error fetching real-time data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    def extract_real_time_results(self, data: dict) -> list:
        results = data.get('results', [])
        return [
            {
                'timestamp': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

    def save_metadata(self, metadata):
        metadata_file = 'metadata_polygon.csv'
        with open(metadata_file, 'a') as f:
            for entry in metadata:
                f.write(f"{entry['source_url']},{entry['fetch_time']},{entry['status']},{entry['data_size']},{entry['symbol']},{entry['date_range']}\n")
