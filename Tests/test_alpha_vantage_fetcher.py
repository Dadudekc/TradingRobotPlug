# C:\TheTradingRobotPlug\Tests\test_alpha_vantage_fetcher.py

import unittest
from unittest.mock import patch, MagicMock
from Scripts.DataFetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher

class TestAlphaVantageDataFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = AlphaVantageDataFetcher()
        self.sample_data = {
            "Time Series (Daily)": {
                "2023-07-01": {
                    "1. open": "150.00",
                    "2. high": "155.00",
                    "3. low": "148.00",
                    "4. close": "152.00",
                    "5. volume": "1000000"
                }
            }
        }

    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url("AAPL", "2023-01-01", "2023-12-31")
        expected_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=ALPHAVANTAGE_API_KEY&outputsize=full&datatype=json"
        self.assertEqual(url, expected_url)

    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [{
            'date': '2023-07-01',
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 152.0,
            'volume': 1000000
        }]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    def test_fetch_data_for_symbol(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_data
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertEqual(data.index.name, 'date')

    @patch('requests.get')
    def test_fetch_data_for_symbol_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

    @patch('Scripts.DataFetchers.alpha_vantage_fetcher.AlphaVantageDataFetcher.utils', new_callable=MagicMock)
    def test_logging_in_extract_results(self, mock_utils):
        sample_data = {
            "Time Series (Daily)": {
                "2023-07-01": {
                    "1. open": "150.00",
                    "2. high": "155.00",
                    "3. low": "148.00",
                    "4. close": "152.00",
                    "5. volume": "1000000"
                }
            }
        }
        self.fetcher.utils = mock_utils
        self.fetcher.extract_results(sample_data)
        expected_log_message = "AlphaVantage: Extracted results: [{'date': '2023-07-01', 'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000}]"
        self.fetcher.utils.logger.debug.assert_called_once_with(expected_log_message)

if __name__ == '__main__':
    unittest.main()
