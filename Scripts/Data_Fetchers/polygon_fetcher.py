import pandas as pd
import aiohttp
from datetime import datetime
from typing import Optional
from base_fetcher import DataFetcher  # Import DataFetcher from base_fetcher
import os
import sys

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.DataLakeHandler import DataLakeHandler  # Import DataLakeHandler


class PolygonDataFetcher(DataFetcher):
    def __init__(self, data_lake_handler: Optional[DataLakeHandler] = None):
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker', 'C:/TheTradingRobotPlug/data/polygon', 'C:/TheTradingRobotPlug/data/processed_polygon', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/polygon.log', 'Polygon', data_lake_handler)

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    def extract_results(self, data: dict) -> list:
        results = data.get('results', [])
        self.utils.logger.debug(f"Polygon: Extracted results: {results}")
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

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = f"{self.base_url}/{symbol}/range/1/minute/2023-01-01/2023-12-31?apiKey={self.api_key}"
        
        try:
            self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
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
                        self.utils.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                        return df
                    else:
                        self.utils.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.utils.logger.error(f"{self.source}: Error fetching real-time data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
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
