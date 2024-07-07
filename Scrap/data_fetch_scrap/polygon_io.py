# C:\TheTradingRobotPlug\Scripts\Data_Fetch\polygon_io.py

import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from Scripts.Utilities.data_fetch_utils import DataFetchUtils

# Load environment variables from .env file
load_dotenv()

class PolygonDataFetcher:
    def __init__(self, api_key, csv_dir):
        self.api_key = api_key
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)
        self.utils = DataFetchUtils("C:/TheTradingRobotPlug/logs/polygon_data_fetcher.log")
        self.logger = self.utils.logger

    def fetch_data_from_polygon(self, ticker_symbols, start_date=None, end_date=None):
        """
        Fetch historical stock data for multiple symbols using the Polygon.io API.

        Args:
            ticker_symbols (list of str): List of stock symbols to fetch data for.
            start_date (str, optional): Start date for data fetching in YYYY-MM-DD format.
            end_date (str, optional): End date for data fetching in YYYY-MM-DD format. Defaults to current date.

        Returns:
            dict of pd.DataFrame: Dictionary where each key is a symbol and value is a DataFrame containing the fetched stock data.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        data_frames = {}

        for symbol in ticker_symbols:
            csv_file_path = os.path.join(self.csv_dir, f"{symbol}_data.csv")

            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path, index_col=0, parse_dates=['date'])
                data_frames[symbol] = df
                self.logger.info(f"Loaded data for {symbol} from the CSV file.")
            else:
                df = self.fetch_data_for_symbol(symbol, start_date, end_date)
                if df is not None:
                    df.to_csv(csv_file_path, index=True)
                    data_frames[symbol] = df
                    self.logger.info(f"Fetched and saved data for {symbol} to CSV file.")

        return data_frames

    def fetch_data_for_symbol(self, symbol, start_date, end_date):
        """
        Fetch historical stock data for a single symbol using the Polygon.io API.

        Args:
            symbol (str): Stock symbol to fetch data for.
            start_date (str): Start date for data fetching in YYYY-MM-DD format.
            end_date (str): End date for data fetching in YYYY-MM-DD format.

        Returns:
            pd.DataFrame or None: DataFrame containing the fetched stock data or None if the fetch fails.
        """
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])

            if isinstance(results, list):
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.drop(['t'], axis=1, inplace=True)
                return df
            else:
                self.logger.warning(f"Fetched data for {symbol} is not in the expected format.")
                return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch data for {symbol}. Error: {str(e)}")
            return None

if __name__ == "__main__":
    ticker_symbols = ["AAPL", "MSFT"]
    api_key = os.getenv('POLYGON_API_KEY')
    csv_dir = os.path.join(os.path.dirname(__file__), "data")
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    fetcher = PolygonDataFetcher(api_key, csv_dir)
    data_frames = fetcher.fetch_data_from_polygon(ticker_symbols, start_date, end_date)

    # You can now work with the fetched data frames as needed.
