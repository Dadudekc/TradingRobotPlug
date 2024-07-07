import os
import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

# Load environment variables from .env file
load_dotenv()

class DataFetcher:
    def __init__(self, api_key_env_var: str, base_url: str, csv_dir_env_var: str, db_path_env_var: str, log_file_env_var: str, source: str):
        self.utils = DataFetchUtils(os.getenv(log_file_env_var))
        self.api_key = os.getenv(api_key_env_var)
        self.base_url = base_url
        self.source = source

        self.csv_dir = os.getenv(csv_dir_env_var)
        self.db_path = os.getenv(db_path_env_var)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.data_store = DataStore(self.csv_dir, self.db_path)

        if not self.api_key:
            self.utils.logger.error(f"{self.source}: API key not found in environment variables.")
            raise ValueError("API key not found in environment variables.")

    def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> dict:
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = {}

        for symbol in ticker_symbols:
            self.utils.logger.info(f"{self.source}: Fetching data for {symbol} from {start_date} to {end_date}")
            data = self.fetch_data_for_symbol(symbol, start_date, end_date)
            if data is not None:
                all_data[symbol] = data

        return all_data

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol, start_date, end_date)

        try:
            self.utils.logger.debug(f"{self.source}: Request URL: {url}")
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            results = self.extract_results(data)

            if results:
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()  # Ensure the index is sorted
                df['symbol'] = symbol
                filtered_df = df.loc[start_date:end_date]
                self.utils.logger.debug(f"{self.source}: Fetched data for {symbol}: {filtered_df}")
                return filtered_df
            else:
                self.utils.logger.warning(f"{self.source}: Fetched data for {symbol} is not in the expected format.")
                return None
        except requests.RequestException as e:
            self.utils.logger.error(f"{self.source}: Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return None

    def save_data(self, data: pd.DataFrame, symbol: str, overwrite=False) -> None:
        if data.empty:
            self.utils.logger.warning(f"{self.source}: No data to save for symbol: {symbol}")
            return

        file_name = f"{symbol}_data.csv"
        self.data_store.save_to_csv(data, file_name, overwrite)
        self.data_store.save_to_sql(data, f"{symbol}_data")
        self.utils.logger.info(f"{self.source}: Data saved to CSV and SQL for symbol: {symbol}")

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_results(self, data: dict) -> list:
        raise NotImplementedError("This method should be implemented by subclasses.")
