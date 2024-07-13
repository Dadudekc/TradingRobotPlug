import os
import sys
import asyncio
import pandas as pd
import aiohttp
import logging
from datetime import datetime
from typing import Optional

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.base_fetcher import DataFetcher
from Scripts.Utilities.data_store import DataStore

class AlphaVantageDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('ALPHAVANTAGE_API_KEY', 
                         'https://www.alphavantage.co/query', 
                         'C:/TheTradingRobotPlug/data/raw/alpha_vantage', 
                         'C:/TheTradingRobotPlug/data/processed/alpha_vantage', 
                         'C:/TheTradingRobotPlug/data/trading_data.db', 
                         'C:/TheTradingRobotPlug/logs/alpha_vantage.log', 
                         'AlphaVantage')
        self.data_store = DataStore()

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
        self.logger.debug(f"AlphaVantage: Extracted results: {results}")
        return results

    async def async_fetch_data(self, symbol: str) -> pd.DataFrame:
        url = self.construct_api_url(symbol, "2023-01-01", "2023-12-31")
        try:
            self.logger.debug(f"{self.source}: Request URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = self.extract_results(data)
                    if results:
                        df = pd.DataFrame(results)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df['symbol'] = symbol
                        self.logger.debug(f"{self.source}: Fetched data for {symbol}: {df}")
                        return df
                    else:
                        self.logger.warning(f"{self.source}: Data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"{self.source}: Error fetching data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_real_time_data(self, symbol: str) -> pd.DataFrame:
        url = f"{self.base_url}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.api_key}&outputsize=compact&datatype=json"
        
        try:
            self.logger.debug(f"{self.source}: Real-time request URL: {url}")
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
                        self.logger.debug(f"{self.source}: Fetched real-time data for {symbol}: {df}")
                        return df
                    else:
                        self.logger.warning(f"{self.source}: Real-time data for {symbol} is not in the expected format.")
                        return pd.DataFrame()
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"{self.source}: Error fetching real-time data for symbol {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"{self.source}: Unexpected error for symbol {symbol}: {e}")
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
        self.logger.debug(f"AlphaVantage: Extracted real-time results: {results}")
        return results

    async def fetch_data(self, symbols, start_date, end_date) -> dict:
        metadata = []
        results = {}
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                url = self.construct_api_url(symbol, start_date, end_date)
                try:
                    self.logger.debug(f"{self.source}: Request URL: {url}")
                    async with session.get(url) as response:
                        fetch_time = datetime.now().isoformat()
                        status = response.status
                        data_size = len(await response.text())
                        
                        if response.status == 429:
                            await asyncio.sleep(60)
                            continue
                        
                        response.raise_for_status()
                        data = await response.json()
                        results[symbol] = self.extract_results(data)
                        
                        # Collect metadata
                        metadata.append({
                            'source_url': url,
                            'fetch_time': fetch_time,
                            'status': status,
                            'data_size': data_size,
                            'symbol': symbol,
                            'date_range': f"{start_date} to {end_date}"
                        })
                        
                        self.logger.info(f"Data successfully fetched from AlphaVantage for {symbol}")
                except aiohttp.ClientError as err:
                    self.logger.error(f"An error occurred: {err}")
        self.save_metadata(metadata)
        return results

    def save_metadata(self, metadata):
        metadata_file = 'metadata_alpha_vantage.csv'
        with open(metadata_file, 'a') as f:
            for entry in metadata:
                f.write(f"{entry['source_url']},{entry['fetch_time']},{entry['status']},{entry['data_size']},{entry['symbol']},{entry['date_range']}\n")

async def main():
    fetcher = AlphaVantageDataFetcher()
    symbols = ["AAPL", "MSFT", "GOOG"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    data = await fetcher.fetch_data(symbols, start_date, end_date)
    for symbol, df in data.items():
        if not df.empty and fetcher.validate_data(df):
            fetcher.save_data(df, symbol, processed=True, overwrite=True, versioning=True, archive=True)
    
    real_time_data = await fetcher.fetch_real_time_data("AAPL")
    if not real_time_data.empty and fetcher.validate_data(real_time_data):
        fetcher.save_data(real_time_data, "AAPL", processed=True, overwrite=True, versioning=True, archive=True)
    
    print("Data fetching completed.")

if __name__ == "__main__":
    asyncio.run(main())
