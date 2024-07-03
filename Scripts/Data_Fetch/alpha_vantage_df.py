# C:\TheTradingRobotPlug\Scripts\Data_Fetch\alpha_vantage_df
# Scripts\Data_Fetch\alpha_vantage_df

import os
import sys
import requests
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from Utilities.data_store import DataStore
from Utilities.data_fetch_utils import setup_logger, load_config, load_paths, ensure_directory_exists

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.ini
config = load_config()
paths = load_paths(config)

class AlphaVantageDataFetcher:
    def __init__(self, csv_dir: Optional[str] = None, db_path: Optional[str] = None):
        self.logger = setup_logger("AlphaVantageDataFetcher", os.path.join(paths['data_folder'], "logs/alpha_vantage.log"))
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"

        if csv_dir is None:
            csv_dir = paths['data_folder']
        if db_path is None:
            db_path = os.path.join(paths['data_folder'], "trading_data.db")

        self.data_store = DataStore(csv_dir, db_path)

        if not self.api_key:
            self.logger.error("API key not found in environment variables.")
            raise ValueError("API key not found in environment variables.")

    def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> dict:
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = {}

        for symbol in ticker_symbols:
            data = self.fetch_data_for_symbol(symbol, start_date, end_date)
            if data is not None:
                all_data[symbol] = data

        return all_data

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(self.base_url, params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",  # Fetch full-length time series
                "datatype": "json",  # Fetch data in JSON format
            })
            response.raise_for_status()

            data = response.json()
            if "Time Series (Daily)" not in data:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)  # Convert index to datetime
            df = df.sort_index()  # Sort and ensure a monotonic index
            df.index.name = "date"
            df['symbol'] = symbol

            # Perform date-based slicing
            filtered_df = df.loc[start_date:end_date]

            return filtered_df
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for symbol {symbol}: {e}")
            return None

    def validate_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if df.index.min() > start_date or df.index.max() < end_date:
            self.logger.warning(f"Requested date range {start_date} to {end_date} is out of available data range for {df['symbol'].iloc[0]}")
            return False
        return True

    def save_data(self, data: pd.DataFrame, symbol: str, overwrite=False) -> None:
        if data.empty:
            self.logger.warning(f"No data to save for symbol: {symbol}")
            return

        file_name = f"{symbol}_data.csv"
        self.data_store.save_to_csv(data, file_name, overwrite)
        self.data_store.save_to_sql(data, f"{symbol}_data")

if __name__ == "__main__":
    try:
        data_fetcher = AlphaVantageDataFetcher()
        ticker_symbols = ["AAPL"]
        start_date = "2022-01-01"
        end_date = "2022-12-31"

        fetched_data = data_fetcher.fetch_data(ticker_symbols, start_date, end_date)

        if fetched_data:
            for symbol, data in fetched_data.items():
                data_fetcher.save_data(data, symbol, overwrite=True)
            print(f"Data fetched and saved for {ticker_symbols}")
        else:
            print("No data fetched.")
    except Exception as e:
        print(f"An error occurred: {e}")
