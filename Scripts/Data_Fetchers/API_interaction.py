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
            self.logger.warning(f"Rate limit reached. Retrying after {retry_after} seconds... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(retry_after)
            result = await self.async_fetch_data()
            if result is not None:
                return result
        self.logger.error(f"Max retries reached for {self.__class__.__name__}")
        return None

class AlphaVantageAPI(BaseAPI):
    def __init__(self, base_url: str, api_key: str):
        super().__init__(base_url, api_key)

    def _construct_url(self, symbol: str, interval: str) -> str:
        return f"{self.base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full&datatype=json"

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
                    print(f"Data fetched from AlphaVantage for {symbol}: {data}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
                print(f"An error occurred: {err}")
            return None

# Test the AlphaVantageAPI separately
async def test_alpha_vantage_api():
    api = AlphaVantageAPI('https://www.alphavantage.co/query', 'test_api_key')
    data = await api.async_fetch_data("AAPL", "daily")
    print(data)

# Uncomment the line below to test the API directly
# asyncio.run(test_alpha_vantage_api())
