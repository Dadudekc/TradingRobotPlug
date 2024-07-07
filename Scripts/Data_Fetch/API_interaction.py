# C:\TheTradingRobotPlug\Scripts\Data_Fetch\API_interaction.py

import asyncio
import aiohttp
import logging
import os
from typing import Optional
from dotenv import load_dotenv

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
        return f"{self.base_url}/time_series/{interval}/{symbol}?apikey={self.api_key}"

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
        return f"{self.base_url}/open-close/{symbol}/{interval}?apiKey={self.api_key}"

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
        return f"{self.base_url}/data/{symbol}/{interval}?apikey={self.api_key}"

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from NASDAQ for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None


# Define API keys and base URLs using environment variables
alpha_vantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
polygon_io_api_key = os.getenv('POLYGON_API_KEY')
nasdaq_api_key = os.getenv('NASDAQ_API_KEY')

alpha_vantage_base_url = 'https://www.alphavantage.co/query'
polygon_io_base_url = 'https://api.polygon.io/v2'
nasdaq_base_url = 'https://api.nasdaq.com/api'

async def main():
    alpha_vantage = AlphaVantageAPI(alpha_vantage_base_url, alpha_vantage_api_key)
    polygon_io = PolygonIOAPI(polygon_io_base_url, polygon_io_api_key)
    nasdaq = NASDAQAPI(nasdaq_base_url, nasdaq_api_key)

    # Fetch data asynchronously (example)
    data_av = await alpha_vantage.async_fetch_data("AAPL", "daily")
    data_pg = await polygon_io.async_fetch_data("AAPL", "daily")
    data_nd = await nasdaq.async_fetch_data("AAPL", "daily")

    # Handle the fetched data
    if data_av:
        print("AlphaVantage Data:", data_av)
    if data_pg:
        print("Polygon.io Data:", data_pg)
    if data_nd:
        print("NASDAQ Data:", data_nd)

asyncio.run(main())
