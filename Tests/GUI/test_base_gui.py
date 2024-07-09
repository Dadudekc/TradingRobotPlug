import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from dotenv import load_dotenv

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.base_fetcher import DataFetcher

class TestBaseFetcher(DataFetcher):
    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"https://api.example.com/data?symbol={symbol}&start_date={start_date}&end_date={end_date}&apikey={self.api_key}"

    def extract_results(self, data: dict) -> list:
        return data.get("results", [])

class TestDataFetcher(unittest.TestCase):

    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def setUp(self):
        # Ensure the .env file is loaded
        load_dotenv(dotenv_path=os.path.join(project_root, '.env'))
        
        # Mock environment variables
        os.makedirs("test_log_dir", exist_ok=True)
        os.makedirs("test_csv_dir", exist_ok=True)
        
        self.fetcher = TestBaseFetcher('ALPHAVANTAGE_API_KEY', 'https://api.example.com', 'CSV_DIR', 'DB_PATH', 'ALPHA_LOG_FILE', 'TestSource')
        self.sample_data = {
            "results": [
                {'date': '2023-07-01', 'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000}
            ]
        }

    @patch('Scripts.Utilities.data_fetch_utils.DataFetchUtils', new_callable=MagicMock)
    @patch('Scripts.Utilities.data_store.DataStore', new_callable=MagicMock)
    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_initialization(self, mock_data_store, mock_data_fetch_utils):
        fetcher = TestBaseFetcher('ALPHAVANTAGE_API_KEY', 'https://api.example.com', 'CSV_DIR', 'DB_PATH', 'ALPHA_LOG_FILE', 'TestSource')
        self.assertEqual(fetcher.base_url, 'https://api.example.com')
        self.assertEqual(fetcher.api_key, os.getenv('ALPHAVANTAGE_API_KEY'))
        self.assertEqual(fetcher.csv_dir, 'test_csv_dir')
        self.assertEqual(fetcher.db_path, 'test_db_path')
        self.assertEqual(fetcher.source, 'TestSource')

    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_construct_api_url(self):
        url = self.fetcher.construct_api_url('AAPL', '2023-01-01', '2023-12-31')
        expected_url = f"https://api.example.com/data?symbol=AAPL&start_date=2023-01-01&end_date=2023-12-31&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
        self.assertEqual(url, expected_url)

    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_extract_results(self):
        results = self.fetcher.extract_results(self.sample_data)
        expected_results = [
            {'date': '2023-07-01', 'open': 150.0, 'high': 155.0, 'low': 148.0, 'close': 152.0, 'volume': 1000000}
        ]
        self.assertEqual(results, expected_results)

    @patch('requests.get')
    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
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
    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_fetch_data_for_symbol_error(self, mock_get):
        mock_get.side_effect = Exception("Test Exception")
        data = self.fetcher.fetch_data_for_symbol('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

    @patch('Scripts.Utilities.data_store.DataStore.save_to_csv')
    @patch('Scripts.Utilities.data_store.DataStore.save_to_sql')
    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_save_data(self, mock_save_to_sql, mock_save_to_csv):
        df = pd.DataFrame(self.sample_data['results'])
        self.fetcher.save_data(df, 'AAPL')
        mock_save_to_csv.assert_called_once()
        mock_save_to_sql.assert_called_once()

    @patch('Scripts.Utilities.data_store.DataStore.save_to_csv')
    @patch('Scripts.Utilities.data_store.DataStore.save_to_sql')
    @patch.dict(os.environ, {
        "ALPHAVANTAGE_API_KEY": "your_test_api_key",
        "ALPHA_LOG_FILE": "test_log_dir/test_log_file.log",
        "CSV_DIR": "test_csv_dir",
        "DB_PATH": "test_db_path"
    })
    def test_save_data_empty(self, mock_save_to_sql, mock_save_to_csv):
        df = pd.DataFrame()
        self.fetcher.save_data(df, 'AAPL')
        mock_save_to_csv.assert_not_called()
        mock_save_to_sql.assert_not_called()

    def tearDown(self):
        if os.path.exists("test_log_dir/test_log_file.log"):
            try:
                os.remove("test_log_dir/test_log_file.log")
            except PermissionError:
                pass  # Log file is still in use
        if os.path.exists("test_log_dir"):
            try:
                os.rmdir("test_log_dir")
            except OSError:
                pass  # Directory is not empty or in use
        if os.path.exists("test_csv_dir"):
            try:
                os.rmdir("test_csv_dir")
            except OSError:
                pass  # Directory is not empty or in use

if __name__ == '__main__':
    unittest.main()
