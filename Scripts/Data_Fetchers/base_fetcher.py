import aiohttp
import pandas as pd
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timedelta
import os
import sys

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Utilities.DataLakeHandler import DataLakeHandler

class DataFetcher:
    """
    Base class for fetching financial data from APIs.

    Attributes:
        api_key (str): The API key for authentication.
        base_url (str): The base URL for the API.
        raw_csv_dir (str): Directory to store raw CSV files.
        processed_csv_dir (str): Directory to store processed CSV files.
        db_path (str): Path to the SQLite database.
        log_file (str): Path to the log file.
        source (str): The source of the data (e.g., AlphaVantage).
        data_lake_handler (Optional[object]): Handler for storing data in a data lake.
    """
    def __init__(self, api_key, base_url, raw_csv_dir, processed_csv_dir, db_path, log_file, source, data_lake_handler: Optional[object] = None):
        """
        Initializes the DataFetcher with the given parameters.
        
        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API.
            raw_csv_dir (str): Directory to store raw CSV files.
            processed_csv_dir (str): Directory to store processed CSV files.
            db_path (str): Path to the SQLite database.
            log_file (str): Path to the log file.
            source (str): The source of the data (e.g., AlphaVantage).
            data_lake_handler (Optional[object]): Handler for storing data in a data lake.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.raw_csv_dir = raw_csv_dir
        self.processed_csv_dir = processed_csv_dir
        self.db_path = db_path
        self.log_file = log_file
        self.source = source
        self.data_lake_handler = data_lake_handler

        # Ensure directories exist
        os.makedirs(self.raw_csv_dir, exist_ok=True)
        os.makedirs(self.processed_csv_dir, exist_ok=True)

        # Initialize DataStore
        self.data_store = DataStore(self.raw_csv_dir, self.db_path)

        # Initialize logger
        self.logger = logging.getLogger(self.source)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        if not self.api_key:
            self.logger.error(f"{self.source}: API key not found in environment variables.")
            raise ValueError("API key not found in environment variables.")

    async def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical data for multiple ticker symbols asynchronously.

        Args:
            ticker_symbols (List[str]): The list of stock symbols to fetch data for.
            start_date (str): The start date for fetching data. Defaults to "2023-01-01".
            end_date (str): The end date for fetching data. Defaults to yesterday's date.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping symbols to their fetched data as pandas DataFrames.
        """
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = {}
        tasks = [self.fetch_data_for_symbol(symbol, start_date, end_date) for symbol in ticker_symbols]
        results = await asyncio.gather(*tasks)

        for symbol, data in zip(ticker_symbols, results):
            if data is not None:
                all_data[symbol] = data

        return all_data

    async def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a given ticker symbol asynchronously.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (str): The start date for fetching data.
            end_date (str): The end date for fetching data.

        Returns:
            Optional[pd.DataFrame]: The fetched data as a pandas DataFrame, or None if no data was fetched.
        """
        url = self.construct_api_url(symbol)
        timeout = aiohttp.ClientTimeout(total=60)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"{self.source}: Failed to fetch data for {symbol}. Status code: {response.status}")
                        return None

                    data = await response.json(content_type=None)
                    self.logger.debug(f"API response for {symbol}: {data}")

                    results = self.extract_results(data, "Time Series (Daily)")

                    if not results:
                        self.logger.warning(f"{self.source}: Fetched data for {symbol} is not in the expected format. Data: {data}")
                        return None

                    df = pd.DataFrame(results)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()  # Ensure the index is sorted
                    df['symbol'] = symbol

                    filtered_df = df.loc[start_date:end_date]
                    self.logger.debug(f"{self.source}: Fetched data for {symbol}: {filtered_df}")

                    return filtered_df

        except aiohttp.ClientConnectionError as e:
            self.logger.error(f"Connection error for symbol {symbol}: {e}")
        except aiohttp.ContentTypeError as e:
            self.logger.error(f"Unexpected content type for symbol {symbol}: {e}")
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout error for symbol {symbol}")
        except Exception as e:
            self.logger.error(f"Unexpected error for symbol {symbol}: {e}")

        return None

    async def fetch_real_time_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetches real-time data for a given ticker symbol asynchronously.

        Args:
            symbol (str): The stock symbol to fetch real-time data for.

        Returns:
            Optional[pd.DataFrame]: The fetched real-time data as a pandas DataFrame.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_data(self, data: pd.DataFrame, symbol: str, processed=False, overwrite=False, versioning=False, archive=False) -> None:
        """
        Saves fetched data to CSV and SQLite, and optionally uploads to a data lake.

        Args:
            data (pd.DataFrame): The data to save.
            symbol (str): The stock symbol associated with the data.
            processed (bool): Whether the data is processed. Defaults to False.
            overwrite (bool): Whether to overwrite existing files. Defaults to False.
            versioning (bool): Whether to enable versioning. Defaults to False.
            archive (bool): Whether to archive existing files. Defaults to False.
        """
        if data.empty:
            self.logger.warning(f"{self.source}: No data to save for symbol: {symbol}")
            return

        file_dir = self.processed_csv_dir if processed else self.raw_csv_dir
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
        if self.data_lake_handler:
            self.data_lake_handler.upload_file(file_path, f"{symbol}/{file_name}")
            self.logger.info(f"{self.source}: Data saved to CSV, SQL, and uploaded to data lake for symbol: {symbol}")
        else:
            self.logger.info(f"{self.source}: Data saved to CSV and SQL for symbol: {symbol}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validates the fetched data to ensure it contains required columns.

        Args:
            data (pd.DataFrame): The data to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        if data.empty or not all(col in data.columns for col in required_columns):
            self.logger.error(f"{self.source}: Validation failed for data")
            return False
        return True

    def archive_existing_file(self, file_path: str) -> None:
        """
        Archives an existing file by moving it to an archive directory with a timestamp.

        Args:
            file_path (str): The path of the file to archive.
        """
        archive_dir = os.path.join(os.path.dirname(file_path), 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        # Append a timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_file_path = os.path.join(archive_dir, f"{name}_{timestamp}{ext}")

        os.rename(file_path, new_file_path)
        self.logger.info(f"{self.source}: Archived existing file to {new_file_path}")

    def construct_api_url(self, symbol: str) -> str:
        """
        Constructs the API URL for fetching data. This method should be implemented by subclasses.

        Args:
            symbol (str): The stock symbol to fetch data for.

        Returns:
            str: The constructed API URL.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_results(self, data: dict, time_series_key: str) -> List[Dict[str, any]]:
        """
        Extracts results from the fetched data. This method should be implemented by subclasses.

        Args:
            data (dict): The fetched data dictionary.
            time_series_key (str): The key for the time series data in the dictionary.

        Returns:
            List[Dict[str, any]]: A list of dictionaries containing the extracted results.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
