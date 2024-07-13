import pandas as pd
from datetime import datetime
import os
import sys
import asyncio
import aiohttp

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.base_fetcher import DataFetcher

class NasdaqDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('NASDAQ_API_KEY', 
                         'https://dataondemand.nasdaq.com/api/v1/historical', 
                         'C:/TheTradingRobotPlug/data/raw/nasdaq', 
                         'C:/TheTradingRobotPlug/data/processed/nasdaq', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/nasdaq.log', 
                         'Nasdaq')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        url = f"{self.base_url}/{symbol}?apiKey={self.api_key}"
        if start_date:
            url += f"&startDate={start_date}"
        if end_date:
            url += f"&endDate={end_date}"
        return url

    def extract_results(self, data: dict) -> list:
        results = data.get('data', [])
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
        url = f"https://dataondemand.nasdaq.com/api/v1/historical/{symbol}/real-time?apiKey={self.api_key}"
        
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
        results = data.get('data', [])
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

async def main():
    fetcher = NasdaqDataFetcher()
    # Fetch historical data
    data = fetcher.fetch_data(["AAPL"])  # fetch_data is not async
    for symbol, df in data.items():
        if fetcher.validate_data(df):
            fetcher.save_data(df, symbol, overwrite=True, versioning=True, archive=True)
    
    # Fetch real-time data
    real_time_data = await fetcher.fetch_real_time_data("AAPL")  # fetch_real_time_data is async
    if not real_time_data.empty:
        fetcher.save_data(real_time_data, "AAPL", overwrite=True, versioning=True, archive=True)
    
    print("Data fetching completed.")

if __name__ == "__main__":
    asyncio.run(main())
