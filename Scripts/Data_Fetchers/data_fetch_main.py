import os
import sys
from pathlib import Path
import asyncio
import aiohttp
import logging
from typing import Optional
from dotenv import load_dotenv

# Add project root to the Python path for module imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from Scripts.Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher
from Scrap.data_fetch_scrap.nasdaq_fetcher import NasdaqDataFetcher
from Scripts.Data_Fetchers.polygon_fetcher import PolygonDataFetcher
from Scripts.Utilities.config_handling import ConfigManager
from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Utilities.data_store import DataStore

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class BaseAPI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def _construct_url(self, symbol: str, interval: str) -> str:
        raise NotImplementedError

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        raise NotImplementedError

    async def handle_rate_limit(self, retry_after=60, max_retries=5):
        for attempt in range(max_retries):
            await asyncio.sleep(retry_after)
            result = await self.async_fetch_data()
            if result is not None:
                return result
        self.logger.error(f"Max retries reached for {self.__class__.__name__}")
        return None

class AlphaVantageAPI(BaseAPI):
    def _construct_url(self, symbol: str, interval: str) -> str:
        return f"{self.base_url}/query?function=TIME_SERIES_{interval.upper()}&symbol={symbol}&apikey={self.api_key}"

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from AlphaVantage for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None

class PolygonIOAPI(BaseAPI):
    def _construct_url(self, symbol: str, interval: str) -> str:
        return f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{interval}?adjusted=true&apiKey={self.api_key}"

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from Polygon.io for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None

class NASDAQAPI(BaseAPI):
    def _construct_url(self, symbol: str, interval: str) -> str:
        return f"{self.base_url}/quote/{symbol}/chart?assetclass=stocks&fromdate={interval}&limit=1&apikey={self.api_key}"

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        url = self._construct_url(symbol, interval)
        self.logger.info(f"Fetching data from Nasdaq for {symbol} using URL: {url}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    self.logger.info(f"Nasdaq API response status for {symbol}: {response.status}")
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from Nasdaq for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred while fetching data from Nasdaq: {err}")
            return None

# Define API keys and base URLs using environment variables
alpha_vantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
polygon_io_api_key = os.getenv('POLYGON_API_KEY')
nasdaq_api_key = os.getenv('NASDAQ_API_KEY')

alpha_vantage_base_url = 'https://www.alphavantage.co'
polygon_io_base_url = 'https://api.polygon.io'
nasdaq_base_url = 'https://api.nasdaq.com/api'

async def fetch_async_data():
    alpha_vantage = AlphaVantageAPI(alpha_vantage_base_url, alpha_vantage_api_key)
    polygon_io = PolygonIOAPI(polygon_io_base_url, polygon_io_api_key)
    nasdaq = NASDAQAPI(nasdaq_base_url, nasdaq_api_key)

    ticker_symbols = ["AAPL", "MSFT", "GOOG"]
    interval = "daily"

    async_fetchers = [
        alpha_vantage.async_fetch_data(symbol, interval) for symbol in ticker_symbols
    ] + [
        polygon_io.async_fetch_data(symbol, interval) for symbol in ticker_symbols
    ] + [
        nasdaq.async_fetch_data(symbol, interval) for symbol in ticker_symbols
    ]

    results = await asyncio.gather(*async_fetchers, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"Error occurred: {result}")
        else:
            print(f"Fetched data: {result}")

def main():
    # Initialize configuration manager
    config = ConfigManager(config_file='config.ini')

    # Fetch API keys from environment variables
    alpha_vantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    nasdaq_api_key = os.getenv('NASDAQ_API_KEY')
    polygon_api_key = os.getenv('POLYGON_API_KEY')

    # Initialize data fetchers
    alpha_vantage_fetcher = AlphaVantageDataFetcher()
    nasdaq_fetcher = NasdaqDataFetcher()
    polygon_fetcher = PolygonDataFetcher()

    # Define the ticker symbols and date range for data fetching
    ticker_symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Fetch and store data for Alpha Vantage
    print("Fetching data from Alpha Vantage...")
    alpha_data = alpha_vantage_fetcher.fetch_data(ticker_symbols, start_date, end_date)
    if alpha_data:
        for symbol, data in alpha_data.items():
            alpha_vantage_fetcher.save_data(data, symbol, overwrite=True)
        print(f"Alpha Vantage data fetched and saved for: {', '.join(alpha_data.keys())}")
    else:
        print("No data fetched from Alpha Vantage.")

    # Fetch and store data for Nasdaq
    print("Fetching data from Nasdaq...")
    nasdaq_data = nasdaq_fetcher.fetch_data(ticker_symbols, start_date, end_date)
    if nasdaq_data:
        for symbol, data in nasdaq_data.items():
            nasdaq_fetcher.save_data(data, symbol, overwrite=True)
        print(f"Nasdaq data fetched and saved for: {', '.join(nasdaq_data.keys())}")
    else:
        print("No data fetched from Nasdaq.")

    # Fetch and store data for Polygon
    print("Fetching data from Polygon...")
    polygon_data = polygon_fetcher.fetch_data(ticker_symbols, start_date, end_date)
    if polygon_data:
        for symbol, data in polygon_data.items():
            polygon_fetcher.save_data(data, symbol, overwrite=True)
        print(f"Polygon data fetched and saved for: {', '.join(polygon_data.keys())}")
    else:
        print("No data fetched from Polygon.")

    # List all saved CSV files
    data_store = DataStore()
    csv_files = data_store.list_csv_files()
    print(f"Available CSV files: {csv_files}")

if __name__ == "__main__":
    main()
    asyncio.run(fetch_async_data())