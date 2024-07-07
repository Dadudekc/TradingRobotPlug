# C:\TheTradingRobotPlug\Scripts\Data_Fetch\nasdaq.py

import logging
import requests
import pandas as pd
import os
import sys

# Ensure the project root is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Scripts'))
sys.path.append(os.path.join(project_root, 'Scripts', 'Utilities'))

from Utilities.data_store import DataStore
from Utilities.data_fetch_utils import DataFetchUtils

# Setup logging
utils = DataFetchUtils("C:/TheTradingRobotPlug/logs/nasdaq.log")
logger = utils.logger

def fetch_data_from_nasdaq(ticker_symbols, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for specific symbols using the NASDAQ API.

    Parameters:
        ticker_symbols (list): List of stock symbols to fetch data for.
        api_key (str): API key for NASDAQ.
        csv_dir (str): Directory to save the fetched data as CSV files.
        start_date (str, optional): Start date for data retrieval (YYYY-MM-DD).
        end_date (str, optional): End date for data retrieval (YYYY-MM-DD).

    Returns:
        list: List of file paths of the saved CSV files.
    """
    data_store = DataStore(csv_dir)
    file_paths = []

    for symbol in ticker_symbols:
        symbol = symbol.strip()
        url = construct_api_url(symbol, api_key, start_date, end_date)

        try:
            response = get_data_from_api(url, symbol)
            if response is None:
                continue

            df = pd.DataFrame(response['data'])
            file_name = f'{symbol}_nasdaq.csv'
            data_store.save_to_csv(df, file_name)
            file_paths.append(os.path.join(csv_dir, file_name))

        except requests.exceptions.RequestException as e:
            handle_request_exception(symbol, e)
        except Exception as e:
            handle_exception(symbol, e)

    return file_paths

def construct_api_url(symbol, api_key, start_date, end_date):
    # Construct API URL based on provided parameters
    url = f'https://dataondemand.nasdaq.com/api/v1/historical/{symbol}?apiKey={api_key}'
    if start_date:
        url += f"&startDate={start_date}"
    if end_date:
        url += f"&endDate={end_date}"
    return url

def get_data_from_api(url, symbol):
    # Make API request and handle errors
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        if 'data' not in data:
            logger.warning(f"No time series data found for symbol {symbol} using NASDAQ API")
            return None
        return data
    except requests.exceptions.RequestException as e:
        handle_request_exception(symbol, e)
        return None

def handle_request_exception(symbol, error):
    # Handle request exceptions and log error
    logger.error(f"Error fetching data for symbol {symbol} using NASDAQ API: {error}")

def handle_exception(symbol, error):
    # Handle unexpected exceptions and log error
    logger.error(f"An unexpected error occurred for symbol {symbol}: {error}")

if __name__ == "__main__":
    try:
        # Example usage
        ticker_symbols = ["AAPL", "MSFT"]
        api_key = os.getenv('NASDAQ_API_KEY')  # Ensure you have set your NASDAQ API key in the environment variables
        csv_dir = 'C:/TheTradingRobotPlug/data/nasdaq'

        fetched_files = fetch_data_from_nasdaq(ticker_symbols, api_key, csv_dir)

        if fetched_files:
            print(f"Data fetched and saved for {ticker_symbols}")
        else:
            print("No data fetched.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
