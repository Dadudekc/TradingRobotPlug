import pandas as pd
from datetime import datetime
import os
import sys
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="C:/TheTradingRobotPlug/.env")

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

    def construct_api_url(self, symbol: str, start_date: str = None, end_date: str = None) -> str:
        url = f"{self.base_url}/{symbol}?apiKey={self.api_key}"
        if start_date:
            url += f"&startDate={start_date}"
        if end_date:
            url += f"&endDate={end_date}"
        print(f"Constructed URL: {url}")  # Debugging step
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
        # Use the appropriate real-time data endpoint
        url = f"https://dataondemand.nasdaq.com/api/v1/last-trade/{symbol}?apiKey={self.api_key}"
        print(f"Real-time data URL: {url}")  # Debugging step

        try:
            self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 404:
                        self.utils.logger.error(f"404 Error for URL: {url}. Check if the endpoint is correct.")
                        print(f"404 Error for URL: {url}. Check if the endpoint is correct.")  # Debugging step
                    elif response.status != 200:
                        self.utils.logger.error(f"Error {response.status} for URL: {url}. Response: {await response.text()}")
                        print(f"Error {response.status} for URL: {url}. Response: {await response.text()}")  # Debugging step
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
                'price': result['price'],
                'volume': result['size']
            }
            for result in results
        ]

async def main():
    fetcher = NasdaqDataFetcher()
    
    # Fetch historical data
    data = fetcher.fetch_data(["AAPL"], start_date="2022-01-01", end_date="2022-12-31")  # fetch_data is not async
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
