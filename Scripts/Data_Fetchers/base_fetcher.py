import os
import sys
import aiohttp
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
load_dotenv()

class DataFetcher:
    def __init__(self, api_key_env_var: str, base_url: str, csv_dir: str, db_path: str, log_file: str, source: str):
        self.utils = DataFetchUtils(log_file)
        self.api_key = os.getenv(api_key_env_var)
        self.base_url = base_url
        self.source = source

        self.csv_dir = csv_dir
        self.db_path = db_path
        os.makedirs(self.csv_dir, exist_ok=True)

        self.data_store = DataStore(self.csv_dir, self.db_path)

        if not self.api_key:
            self.utils.logger.error(f"{self.source}: API key not found in environment variables.")
            raise ValueError("API key not found in environment variables.")

    async def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> dict:
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = {}

        for symbol in ticker_symbols:
            self.utils.logger.info(f"{self.source}: Fetching data for {symbol} from {start_date} to {end_date}")
            data = await self.fetch_data_for_symbol(symbol, start_date, end_date)
            if data is not None:
                all_data[symbol] = data

        return all_data

    async def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol, start_date, end_date)

        try:
            self.utils.logger.debug(f"{self.source}: Request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
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
        except aiohttp.ClientResponseError as e:
            self.utils.logger.error(f"{self.source}: Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return None

    def save_data(self, data: pd.DataFrame, symbol: str, overwrite=False, versioning=False, archive=False) -> None:
        if data.empty:
            self.utils.logger.warning(f"{self.source}: No data to save for symbol: {symbol}")
            return

        file_name = f"{symbol}_data.csv"
        file_path = os.path.join(self.csv_dir, file_name)
        
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
                self.utils.logger.warning(f"{self.source}: {file_path} already exists. Skipping save. Set overwrite=True to overwrite it.")
                return

        self.data_store.save_to_csv(data, file_path, overwrite)
        self.data_store.save_to_sql(data, f"{symbol}_data")
        self.utils.logger.info(f"{self.source}: Data saved to CSV and SQL for symbol: {symbol}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ["open", "high", "low", "close", "volume"]
        if data.empty or not all(col in data.columns for col in required_columns):
            self.utils.logger.error(f"{self.source}: Validation failed for data")
            return False
        return True

    def archive_existing_file(self, file_path: str) -> None:
        archive_dir = os.path.join(self.csv_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(archive_dir, base_name)
        os.rename(file_path, new_file_path)
        self.utils.logger.info(f"{self.source}: Archived existing file to {new_file_path}")

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_results(self, data: dict) -> list:
        raise NotImplementedError("This method should be implemented by subclasses.")
