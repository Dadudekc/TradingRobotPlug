#C:\TheTradingRobotPlug\Scripts\Data_Fetchers\base_fetcher.py

import os
import sys
import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

# Load environment variables from .env file
load_dotenv(dotenv_path="C:/TheTradingRobotPlug/.env")

class DataFetcher:
    def __init__(self, api_key_env_var: str, base_url: str, raw_csv_dir: str, processed_csv_dir: str, db_path: str, log_file: str, source: str):
        self.utils = DataFetchUtils(log_file)
        self.logger = self.utils.logger  # Initialize the logger
        self.api_key = os.getenv(api_key_env_var)
        self.base_url = base_url
        self.source = source

        self.raw_csv_dir = raw_csv_dir
        self.processed_csv_dir = processed_csv_dir
        self.db_path = db_path
        os.makedirs(self.raw_csv_dir, exist_ok=True)
        os.makedirs(self.processed_csv_dir, exist_ok=True)

        self.data_store = DataStore(self.raw_csv_dir, self.db_path)

        if not self.api_key:
            self.logger.error(f"{self.source}: API key not found in environment variables.")
            raise ValueError("API key not found in environment variables.")

    def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> dict:
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = {}

        for symbol in ticker_symbols:
            self.logger.info(f"{self.source}: Fetching data for {symbol} from {start_date} to {end_date}")
            data = self.fetch_data_for_symbol(symbol, start_date, end_date)
            if data is not None:
                all_data[symbol] = data

        return all_data

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol, start_date, end_date)

        try:
            self.logger.debug(f"{self.source}: Request URL: {url}")
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
                self.logger.debug(f"{self.source}: Fetched data for {symbol}: {filtered_df}")
                return filtered_df
            else:
                self.logger.warning(f"{self.source}: Fetched data for {symbol} is not in the expected format.")
                return None
        except requests.RequestException as e:
            self.logger.error(f"{self.source}: Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return None

    def save_data(self, data: pd.DataFrame, symbol: str, processed=False, overwrite=False, versioning=False, archive=False) -> None:
        if data.empty:
            self.logger.warning(f"{self.source}: No data to save for symbol: {symbol}")
            return

        if processed:
            file_dir = self.processed_csv_dir
        else:
            file_dir = self.raw_csv_dir
        
        file_name = f"{symbol}_data.csv"
        file_path = os.path.join(file_dir, file_name)
        
        if os.path.exists(file_path):
            if archive:
                self.archive_existing_file(file_path)
            if versioning:
                version = 1
                new_file_path = file_path.replace(".csv", f"_v{version}.csv")
                while os.path.exists(new_file_path):
                    version += 1
                    new_file_path = file_path.replace(".csv", f"_v{version}.csv")
                file_path = new_file_path
            elif not overwrite:
                self.logger.warning(f"{self.source}: {file_path} already exists. Skipping save. Set overwrite=True to overwrite it.")
                return

        self.data_store.save_to_csv(data, file_path, overwrite)
        self.data_store.save_to_sql(data, f"{symbol}_data")
        self.logger.info(f"{self.source}: Data saved to CSV and SQL for symbol: {symbol}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        # Example validation: Check if data is non-empty and has expected columns
        required_columns = ["open", "high", "low", "close", "volume"]
        if data.empty or not all(col in data.columns for col in required_columns):
            self.logger.error(f"{self.source}: Validation failed for data")
            return False
        return True

    def archive_existing_file(self, file_path: str) -> None:
        archive_dir = os.path.join(os.path.dirname(file_path), 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        
        # Append a timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_file_path = os.path.join(archive_dir, f"{name}_{timestamp}{ext}")
        
        os.rename(file_path, new_file_path)
        self.logger.info(f"{self.source}: Archived existing file to {new_file_path}")

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_results(self, data: dict) -> list:
        raise NotImplementedError("This method should be implemented by subclasses.")
