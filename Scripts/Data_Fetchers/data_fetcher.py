import os
import sys
import asyncio
import pandas as pd
import aiohttp
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Utilities.DataLakeHandler import DataLakeHandler

# Load environment variables from .env file
load_dotenv()

class DataFetcher:
    def __init__(self, api_key_env_var: str, base_url: str, csv_dir: str, db_path: str, log_file: str, source: str, data_lake_handler: DataLakeHandler):
        self.utils = DataFetchUtils(log_file)
        self.api_key = os.getenv(api_key_env_var)
        self.base_url = base_url
        self.source = source
        self.data_lake_handler = data_lake_handler

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
        tasks = [self.fetch_data_for_symbol(symbol, start_date, end_date) for symbol in ticker_symbols]
        results = await asyncio.gather(*tasks)

        for symbol, data in zip(ticker_symbols, results):
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

    async def fetch_real_time_data(self, symbol: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError("This method should be implemented by subclasses.")

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
        self.data_lake_handler.upload_file(file_path, f"{symbol}/{file_name}")
        self.utils.logger.info(f"{self.source}: Data saved to CSV, SQL, and uploaded to S3 for symbol: {symbol}")

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

class AlphaVantageDataFetcher(DataFetcher):
    def __init__(self, data_lake_handler):
        super().__init__('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/query', 'C:/TheTradingRobotPlug/data/alpha_vantage', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/alpha_vantage.log', 'AlphaVantage', data_lake_handler)

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

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = f"{self.base_url}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.api_key}&outputsize=compact&datatype=json"
        
        try:
            self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = self.extract_real_time_results(data)
                    
                    if results:
                        df = pd.DataFrame(results)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        df['symbol'] = symbol
                        self.utils.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                        return df
                    else:
                        self.utils.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.utils.logger.error(f"{self.source}: Error fetching real-time data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    def extract_real_time_results(self, data: dict) -> list:
        time_series = data.get("Time Series (1min)", {})
        results = [
            {
                'timestamp': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]
        self.utils.logger.debug(f"AlphaVantage: Extracted real-time results: {results}")
        return results

class PolygonDataFetcher(DataFetcher):
    def __init__(self, data_lake_handler):
        super().__init__('POLYGON_API_KEY', 'https://api.polygon.io/v2/aggs/ticker', 'C:/TheTradingRobotPlug/data/polygon', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/polygon.log', 'Polygon', data_lake_handler)

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

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = f"{self.base_url}/{symbol}/range/1/minute/2023-01-01/2023-12-31?apiKey={self.api_key}"
        
        try:
            self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = self.extract_real_time_results(data)
                    
                    if results:
                        df = pd.DataFrame(results)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        df['symbol'] = symbol
                        self.utils.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                        return df
                    else:
                        self.utils.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.utils.logger.error(f"{self.source}: Error fetching real-time data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    def extract_real_time_results(self, data: dict) -> list:
        results = data.get('results', [])
        return [
            {
                'timestamp': datetime.utcfromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v']
            }
            for result in results
        ]

# Usage example
async def main():
    data_lake_handler = DataLakeHandler(bucket_name='your-s3-bucket-name')
    
    fetchers = [
        AlphaVantageDataFetcher(data_lake_handler),
        PolygonDataFetcher(data_lake_handler)
    ]
    symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"

    for fetcher in fetchers:
        data = await fetcher.fetch_data(symbols, start_date, end_date)
        for symbol, df in data.items():
            if df is not None and not df.empty and fetcher.validate_data(df):
                fetcher.save_data(df, symbol)
                print(f"Data for {symbol} from {fetcher.source}:\n{df}")

if __name__ == "__main__":
    asyncio.run(main())
