import os
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import time 

# Add project root to the Python path for module imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from Scripts.Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher
from Scripts.Data_Fetchers.polygon_fetcher import PolygonDataFetcher
from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

# Ensure logging setup
logger = DataFetchUtils("C:/TheTradingRobotPlug/logs/data_fetch_main.log").logger

# Load environment variables from .env file
load_dotenv()

async def fetch_data(symbols, start_date, end_date):
    """
    Asynchronously fetches historical data for multiple symbols from Alpha Vantage and Polygon APIs,
    and saves the data to CSV files.

    Args:
        symbols (list): List of stock symbols to fetch data for.
        start_date (str): Start date for data fetching in 'YYYY-MM-DD' format.
        end_date (str): End date for data fetching in 'YYYY-MM-DD' format.

    Returns:
        list: List of filenames of the fetched and saved CSV files.
    """
    alpha_vantage_fetcher = AlphaVantageDataFetcher()
    polygon_fetcher = PolygonDataFetcher()
    data_store = DataStore()
    fetched_files = []

    async def fetch_and_save(fetcher, symbol):
        """
        Fetches data for a given symbol using the specified fetcher and saves it to a CSV file.

        Args:
            fetcher: The data fetcher object (AlphaVantageDataFetcher or PolygonDataFetcher).
            symbol (str): The stock symbol to fetch data for.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame, or None if no data was fetched.
        """
        df = await fetcher.fetch_data_for_symbol(symbol, start_date, end_date)
        if df is not None and not df.empty:
            filename = f"{symbol}_{fetcher.source.lower()}_data_{start_date}_to_{end_date}.csv"
            fetcher.save_data(df, symbol, processed=False, overwrite=True)
            fetched_files.append(filename)
            logger.info(f"{fetcher.source} data fetched and saved for {symbol} as {filename}")
            return df
        return None

    # Fetch data from Alpha Vantage
    logger.info("Fetching data from Alpha Vantage...")
    alpha_data = await asyncio.gather(*[fetch_and_save(alpha_vantage_fetcher, symbol) for symbol in symbols])

    # Check if any data was fetched from Alpha Vantage
    if not any(df is not None and not df.empty for df in alpha_data):
        logger.info("No data fetched from Alpha Vantage. Fetching data from Polygon...")
        polygon_data = await asyncio.gather(*[fetch_and_save(polygon_fetcher, symbol) for symbol in symbols])
        if not any(df is not None and not df.empty for df in polygon_data):
            logger.info("No data fetched from Polygon either.")

    # List all saved CSV files from both raw and processed directories
    csv_files_raw = data_store.list_csv_files()
    csv_files_processed = data_store.list_csv_files(directory=data_store.processed_csv_dir)
    csv_files = csv_files_raw + csv_files_processed
    logger.info(f"Available CSV files: {csv_files}")
    return fetched_files

def fetch_data_from_yfinance(symbol, start_date, end_date):
    """
    Fetches historical data for a given symbol from Yahoo Finance and returns it as a pandas DataFrame.

    Args:
        symbol (str): The stock symbol to fetch data for.
        start_date (str): Start date for data fetching in 'YYYY-MM-DD' format.
        end_date (str): End date for data fetching in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: The fetched data as a pandas DataFrame, or an empty DataFrame if no data was fetched.
    """
    try:
        yf_ticker = yf.Ticker(symbol)
        data = yf_ticker.history(start=start_date, end=end_date)
        if not data.empty:
            data.reset_index(inplace=True)
            data.rename(columns={
                "Date": "date", 
                "Open": "open", 
                "High": "high", 
                "Low": "low", 
                "Close": "close", 
                "Volume": "volume"
            }, inplace=True)
            return data
    except Exception as e:
        logger.error(f"yfinance fetch error: {e}")
    return pd.DataFrame()

async def main(symbols, start_date=None, end_date=None):
    """
    Main function to fetch data for given symbols within the specified date range.

    Args:
        symbols (list): List of stock symbols to fetch data for.
        start_date (str, optional): Start date for data fetching in 'YYYY-MM-DD' format. Defaults to one year ago from today.
        end_date (str, optional): End date for data fetching in 'YYYY-MM-DD' format. Defaults to today's date.

    Returns:
        list: List of filenames of the fetched and saved CSV files.
    """
    # Set default dates to one year from today if not provided
    if not start_date or not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    return await fetch_data(symbols, start_date, end_date)

if __name__ == "__main__":
    # List of stock symbols to fetch data for
    symbols = ["AAPL", "MSFT", "GOOG"]
    # Fetch data and print the filenames of the saved CSV files
    fetched_data = asyncio.run(main(symbols))
    print(fetched_data)
