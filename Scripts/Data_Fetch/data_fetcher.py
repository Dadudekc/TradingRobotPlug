import os
import sys
import requests
import pandas as pd
import logging
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

class AlphaVantageDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/query', 'C:/TheTradingRobotPlug/data/csv', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/alpha_vantage.log', 'AlphaVantage')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full&datatype=json"

    def extract_results(self, data: dict) -> list:
        time_series = data.get("Time Series (Daily)", {})
        results = [
            {
                'date': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]
        self.utils.logger.debug(f"AlphaVantage: Extracted results: {results}")
        return results

class NasdaqDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('NASDAQ_API_KEY', 'https://dataondemand.nasdaq.com/api/v1/historical', 'C:/TheTradingRobotPlug/data/nasdaq', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/nasdaq.log', 'Nasdaq')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        url = f"{self.base_url}/{symbol}?apiKey={self.api_key}"
        if start_date:
            url += f"&startDate={start_date}"
        if end_date:
            url += f"&endDate={end_date}"
        return url

    def extract_results(self, data: dict) -> list:
        results = data.get('data', [])
        self.utils.logger.debug(f"Nasdaq: Extracted results: {results}")
        return [
            {
                'date': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol, start_date, end_date)

        try:
            self.utils.logger.debug(f"Nasdaq: Request URL: {url}")
            response = requests.get(url)
            if response.status_code == 404:
                self.utils.logger.error(f"Nasdaq: 404 Client Error - The requested URL was not found on the server.")
                return None
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
                self.utils.logger.debug(f"Nasdaq: Fetched data for {symbol}: {filtered_df}")
                return filtered_df
            else:
                self.utils.logger.warning(f"Nasdaq: Fetched data for {symbol} is not in the expected format.")
                return None
        except requests.RequestException as e:
            self.utils.logger.error(f"Nasdaq: Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.utils.logger.error(f"Nasdaq: Unexpected error for symbol {symbol}: {e}")
            return None

class PolygonDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker', 'C:/TheTradingRobotPlug/data/polygon', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/polygon_data_fetcher.log', 'Polygon')

    def construct_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        return f"{self.base_url}/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

    def extract_results(self, data: dict) -> list:
        results = data.get('results', [])
        self.utils.logger.debug(f"Polygon: Extracted results: {results}")
        return [
            {
                'date': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        url = self.construct_api_url(symbol, start_date, end_date)

        try:
            self.utils.logger.debug(f"Polygon: Request URL: {url}")
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
                self.utils.logger.debug(f"Polygon: Fetched data for {symbol}: {filtered_df}")
                return filtered_df
            else:
                self.utils.logger.warning(f"Polygon: Fetched data for {symbol} is not in the expected format.")
                return None
        except requests.RequestException as e:
            self.utils.logger.error(f"Polygon: Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.utils.logger.error(f"Polygon: Unexpected error for symbol {symbol}: {e}")
            return None


if __name__ == "__main__":
    data_fetchers = [
        AlphaVantageDataFetcher(),
        NasdaqDataFetcher(),
        PolygonDataFetcher()
    ]
    
    ticker_symbols = ["AAPL"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"

    for fetcher in data_fetchers:
        fetched_data = fetcher.fetch_data(ticker_symbols, start_date, end_date)

        if fetched_data:
            for symbol, data in fetched_data.items():
                fetcher.save_data(data, symbol, overwrite=True)
            print(f"Data fetched and saved for {ticker_symbols} from {fetcher.source}")
        else:
            print(f"No data fetched for {ticker_symbols} from {fetcher.source}.")
