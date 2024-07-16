# C:\TheTradingRobotPlug\Tests\Data_Fetch\test_api_interaction.py

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
        self.api = AlphaVantageAPI(self.base_url)

    @patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
    async def test_async_fetch_data_success(self, mock_get):
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__ = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "some data"})
        mock_get.return_value = mock_response

        data = await self.api.async_fetch_data("AAPL", "DAILY")
        self.assertEqual(data, {"data": "some data"})

    @patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
    async def test_async_fetch_data_rate_limit(self, mock_get):
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__ = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value={"Note": "API call frequency is 5 calls per minute and 500 calls per day."})
        mock_get.return_value = mock_response

        with patch.object(self.api, 'handle_rate_limit', AsyncMock(return_value={"Note": "API call frequency is 5 calls per minute and 500 calls per day."})):
            data = await self.api.async_fetch_data("AAPL", "DAILY")
            self.assertIn("Note", data)

    @patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
    async def test_async_fetch_data_client_error(self, mock_get):
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__ = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"Error Message": "Invalid API call"})
        mock_get.return_value = mock_response

        data = await self.api.async_fetch_data("AAPL", "DAILY")
        self.assertIn("Error Message", data)

if __name__ == '__main__':
    unittest.main()
