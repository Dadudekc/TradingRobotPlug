import os
import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv

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

    # Fetch and store data for Alpha Vantage
    print("Fetching data from Alpha Vantage...")
    alpha_data = await alpha_vantage_fetcher.async_fetch_data(symbols, start_date, end_date)
    if alpha_data:
        for symbol, data in alpha_data.items():
            alpha_vantage_fetcher.save_data(data, symbol, overwrite=True)
        print(f"Alpha Vantage data fetched and saved for: {', '.join(alpha_data.keys())}")
    else:
        print("No data fetched from Alpha Vantage.")

    # Fetch and store data for Polygon
    print("Fetching data from Polygon...")
    polygon_data = await polygon_fetcher.async_fetch_data(symbols, start_date, end_date)
    if polygon_data:
        for symbol, data in polygon_data.items():
            polygon_fetcher.save_data(data, symbol, overwrite=True)
        print(f"Polygon data fetched and saved for: {', '.join(polygon_data.keys())}")
    else:
        print("No data fetched from Polygon.")

    # List all saved CSV files
    data_store = DataStore()
    csv_files = data_store.list_csv_files()
    print(f"Available CSV files: {csv_files}")

async def main(symbols, start_date, end_date):
    await fetch_data(symbols, start_date, end_date)

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    asyncio.run(main(symbols, start_date, end_date))
