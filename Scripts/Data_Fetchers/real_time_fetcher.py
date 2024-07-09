# C:\TheTradingRobotPlug\Scripts\Data_Fetchers\real_time_fetcher.py

import os
import sys
import requests
import pandas as pd
from typing import Optional
from datetime import datetime

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.data_fetcher import DataFetcher

class RealTimeDataFetcherMixin:
    def fetch_real_time_data(self, ticker_symbol: str) -> Optional[pd.DataFrame]:
        url = self.construct_real_time_api_url(ticker_symbol)
        
        try:
            self.utils.logger.debug(f"{self.source}: Real-time request URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            results = self.extract_real_time_results(data)
            
            if results:
                df = pd.DataFrame(results)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df['symbol'] = ticker_symbol
                self.utils.logger.debug(f"{self.source}: Fetched real-time data for {ticker_symbol}: {df}")
                return df
            else:
                self.utils.logger.warning(f"{self.source}: Real-time data for {ticker_symbol} is not in the expected format.")
                return None
        except requests.RequestException as e:
            self.utils.logger.error(f"{self.source}: Error fetching real-time data for symbol {ticker_symbol}: {e}")
            return None
        except Exception as e:
            self.utils.logger.error(f"{self.source}: Unexpected error for symbol {ticker_symbol}: {e}")
            return None

    def construct_real_time_api_url(self, symbol: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_real_time_results(self, data: dict) -> list:
        raise NotImplementedError("This method should be implemented by subclasses.")
