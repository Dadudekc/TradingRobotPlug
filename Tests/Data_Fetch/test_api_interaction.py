import os
import sys
import unittest
from unittest.mock import patch, AsyncMock
import asyncio

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.API_interaction import AlphaVantageAPI

class TestAlphaVantageAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = "YOUR_API_KEY"
        self.api = AlphaVantageAPI(self.base_url, self.api_key)

    @patch('aiohttp.ClientSession')
    def test_async_fetch_data_success(self, MockClientSession):
        async def run_test():
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__ = AsyncMock()
            mock_response.json = AsyncMock(return_value={"data": "some data"})
            mock_session.get.return_value = mock_response
            MockClientSession.return_value.__aenter__.return_value = mock_session
            MockClientSession.return_value.__aexit__ = AsyncMock()

            data = await self.api.async_fetch_data("AAPL", "DAILY")
            self.assertEqual(data, {"data": "some data"})
        
        asyncio.run(run_test())

    @patch('aiohttp.ClientSession')
    def test_async_fetch_data_rate_limit(self, MockClientSession):
        async def run_test():
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__ = AsyncMock()
            mock_response.json = AsyncMock(return_value={"Note": "API call frequency is 5 calls per minute and 500 calls per day."})
            mock_session.get.return_value = mock_response
            MockClientSession.return_value.__aenter__.return_value = mock_session
            MockClientSession.return_value.__aexit__ = AsyncMock()

            data = await self.api.async_fetch_data("AAPL", "DAILY")
            self.assertIn("Note", data)
        
        asyncio.run(run_test())

    @patch('aiohttp.ClientSession')
    def test_async_fetch_data_client_error(self, MockClientSession):
        async def run_test():
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__ = AsyncMock()
            mock_response.json = AsyncMock(return_value={"Error Message": "Invalid API call"})
            mock_session.get.return_value = mock_response
            MockClientSession.return_value.__aenter__.return_value = mock_session
            MockClientSession.return_value.__aexit__ = AsyncMock()

            data = await self.api.async_fetch_data("AAPL", "DAILY")
            self.assertIn("Error Message", data)
        
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()

