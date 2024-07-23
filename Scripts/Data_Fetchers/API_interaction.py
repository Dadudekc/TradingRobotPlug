# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\API_interaction.py

import asyncio
import aiohttp
import logging
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class BaseAPI:
    """
    A base class for interacting with APIs.

    Attributes:
        base_url (str): The base URL of the API.
        api_key (str): The API key for authentication.
        logger (logging.Logger): Logger for logging messages.
    """
    def __init__(self, base_url: str):
        """
        Initializes the BaseAPI with the given base URL.
        
        Args:
            base_url (str): The base URL of the API.
        """
        self.base_url = base_url
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.logger = logging.getLogger(self.__class__.__name__)

    def _construct_url(self, symbol: str, interval: str) -> str:
        """
        Constructs the URL for API requests.

        Args:
            symbol (str): The stock symbol to fetch data for.
            interval (str): The interval for data fetching.

        Returns:
            str: The constructed API URL.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        """
        Asynchronously fetches data from the API.

        Args:
            symbol (str): The stock symbol to fetch data for.
            interval (str): The interval for data fetching.

        Returns:
            Optional[dict]: The fetched data as a dictionary, or None if fetching fails.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    async def handle_rate_limit(self, retry_after=60, max_retries=5):
        """
        Handles rate limiting by retrying the request after a delay.

        Args:
            retry_after (int): The delay in seconds before retrying. Defaults to 60.
            max_retries (int): The maximum number of retries. Defaults to 5.

        Returns:
            Optional[dict]: The fetched data as a dictionary, or None if all retries fail.
        """
        for attempt in range(max_retries):
            self.logger.warning(f"Rate limit reached. Retrying after {retry_after} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(retry_after)
            result = await self.async_fetch_data()
            if result is not None:
                return result
        self.logger.error(f"Max retries reached for {self.__class__.__name__}")
        return None

class AlphaVantageAPI(BaseAPI):
    """
    A class for interacting with the Alpha Vantage API.

    Inherits from BaseAPI.
    """
    def __init__(self, base_url: str):
        """
        Initializes the AlphaVantageAPI with the given base URL.

        Args:
            base_url (str): The base URL of the Alpha Vantage API.
        """
        super().__init__(base_url)

    def _construct_url(self, symbol: str, interval: str) -> str:
        """
        Constructs the URL for Alpha Vantage API requests.

        Args:
            symbol (str): The stock symbol to fetch data for.
            interval (str): The interval for data fetching.

        Returns:
            str: The constructed API URL.
        """
        return f"{self.base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full&datatype=json"

    async def async_fetch_data(self, symbol: str, interval: str) -> Optional[dict]:
        """
        Asynchronously fetches data from the Alpha Vantage API.

        Args:
            symbol (str): The stock symbol to fetch data for.
            interval (str): The interval for data fetching.

        Returns:
            Optional[dict]: The fetched data as a dictionary, or None if fetching fails.
        """
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

# Test the AlphaVantageAPI separately
async def test_alpha_vantage_api():
    """
    Tests the AlphaVantageAPI by fetching data for a given symbol.
    """
    api = AlphaVantageAPI('https://www.alphavantage.co/query')
    data = await api.async_fetch_data("AAPL", "daily")
    print(data)

# Uncomment the line below to test the API directly
# asyncio.run(test_alpha_vantage_api())
