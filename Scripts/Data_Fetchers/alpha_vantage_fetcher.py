import os
import sys
import asyncio
import pandas as pd
import aiohttp

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.base_fetcher import DataFetcher

class AlphaVantageDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__('ALPHAVANTAGE_API_KEY', 'https://www.alphavantage.co/query', 'C:/TheTradingRobotPlug/data/alpha_vantage', 'C:/TheTradingRobotPlug/data/trading_data.db', 'C:/TheTradingRobotPlug/logs/alpha_vantage.log', 'AlphaVantage')

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

async def main():
    fetcher = AlphaVantageDataFetcher()
    data = await fetcher.fetch_data(["AAPL"])
    for symbol, df in data.items():
        if fetcher.validate_data(df):
            fetcher.save_data(df, symbol, overwrite=True, versioning=True, archive=True)  # Use versioning and archiving
    print("Data fetching completed.")

if __name__ == "__main__":
    asyncio.run(main())
