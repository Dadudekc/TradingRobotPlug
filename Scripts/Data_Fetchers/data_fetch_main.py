import os
import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add project root to the Python path for module imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from Scripts.Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher
from Scripts.Data_Fetchers.polygon_fetcher import PolygonDataFetcher
from Scripts.Utilities.data_store import DataStore

# Load environment variables from .env file
load_dotenv()

async def fetch_data(symbols, start_date, end_date):
    alpha_vantage_fetcher = AlphaVantageDataFetcher()
    polygon_fetcher = PolygonDataFetcher()
    data_store = DataStore()
    fetched_files = []

    # Fetch and store data for Alpha Vantage
    print("Fetching data from Alpha Vantage...")
    alpha_data = {}
    for symbol in symbols:
        df = await alpha_vantage_fetcher.async_fetch_data(symbol)
        if not df.empty:
            filename = f"{symbol}_alpha_vantage_data_{start_date}_to_{end_date}.csv"
            alpha_vantage_fetcher.save_data(df, filename, overwrite=True)
            alpha_data[symbol] = df
            fetched_files.append(filename)
            print(f"Alpha Vantage data fetched and saved for {symbol} as {filename}")
    if not alpha_data:
        print("No data fetched from Alpha Vantage.")

        # If no data from Alpha Vantage, use Polygon
        print("Fetching data from Polygon...")
        polygon_data = {}
        for symbol in symbols:
            df = await polygon_fetcher.async_fetch_data(symbol, start_date, end_date)
            if not df.empty:
                filename = f"{symbol}_polygon_data_{start_date}_to_{end_date}.csv"
                polygon_fetcher.save_data(df, filename, overwrite=True)
                polygon_data[symbol] = df
                fetched_files.append(filename)
                print(f"Polygon data fetched and saved for {symbol} as {filename}")
        if not polygon_data:
            print("No data fetched from Polygon either.")

    # List all saved CSV files
    csv_files = data_store.list_csv_files()
    print(f"Available CSV files: {csv_files}")
    return fetched_files

async def main(symbols, start_date=None, end_date=None):
    # Set default dates to one year from today if not provided
    if not start_date or not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    return await fetch_data(symbols, start_date, end_date)

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG"]
    asyncio.run(main(symbols))
