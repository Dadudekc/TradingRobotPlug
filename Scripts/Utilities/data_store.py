# C:\TheTradingRobotPlug\Scripts\Utilities\data_store.py

import os
import pandas as pd
import pickle
import sys
from pathlib import Path
import talib  # Ensure TA-Lib is imported


# Add project root to the Python path for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))
    
from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Utilities.config_handling import ConfigManager

class DataStore:
    def __init__(self, csv_dir='C:/TheTradingRobotPlug/data/alpha_vantage', db_path='C:/TheTradingRobotPlug/data/trading_data.db'):
        self.csv_dir = Path(csv_dir)
        self.processed_csv_dir = self.csv_dir / 'processed'
        self.raw_csv_dir = self.csv_dir / 'raw'
        self.db_path = Path(db_path)
        self.data = {}
        self.utils = DataFetchUtils("C:/TheTradingRobotPlug/logs/data_store.log")
        self.utils.ensure_directory_exists(self.csv_dir)
        self.utils.ensure_directory_exists(self.processed_csv_dir)
        self.utils.ensure_directory_exists(self.raw_csv_dir)
        self.config_manager = ConfigManager()  # Assuming you will use this for some configuration settings

    def add_data(self, ticker, data):
        if not data:
            raise ValueError("Data cannot be empty")
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        self.data[ticker] = data
        self.utils.logger.info(f"Added data for {ticker}")

    def get_data(self, ticker):
        return self.data.get(ticker, None)

    def save_store(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.data, file)
        self.utils.logger.info(f"Saved data store to {file_path}")

    def load_store(self, file_path):
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file)
        self.utils.logger.info(f"Loaded data store from {file_path}")

    def save_to_csv(self, df, file_name, overwrite=False):
        file_path = self.csv_dir / file_name
        # Ensure 'date' is a column before saving
        if df.index.name == 'date':
            df = df.reset_index()
        df.to_csv(file_path, index=False)
        self.utils.logger.info(f"Data saved to CSV at {file_path}")

    def save_to_sql(self, df, table_name, if_exists='replace'):
        self.utils.save_data_to_sql(df, table_name, self.db_path, if_exists)

    def fetch_from_csv(self, file_name):
        file_path = self.csv_dir / file_name
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            self.utils.logger.info(f"Fetched data from CSV at {file_path}")
            self.utils.logger.info(f"DataFrame columns: {df.columns}")
            self.utils.logger.info(f"DataFrame head: \n{df.head()}")
            return df
        except FileNotFoundError as e:
            self.utils.logger.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            self.utils.logger.error(f"Error fetching data from CSV: {e}")
            raise e

    def fetch_from_sql(self, table_name):
        return self.utils.fetch_data_from_sql(table_name, self.db_path)

    def list_csv_files(self, directory=None):
        if directory is None:
            directory = self.csv_dir
        return [f.name for f in directory.glob('*.csv')]

    def load_data(self, symbol):
        file_path = self.csv_dir / f'{symbol}_data.csv'
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                self.utils.logger.info(f"Loaded data for {symbol} from {file_path}")
                self.utils.logger.info(f"DataFrame columns: {df.columns}")
                self.utils.logger.info(f"DataFrame head: \n{df.head()}")
                return df
            except Exception as e:
                self.utils.logger.error(f"Error loading data for {symbol} from {file_path}: {e}")
                return None
        else:
            self.utils.logger.warning(f"No data found for {symbol} at {file_path}")
            return None

    def save_data(self, data, symbol, processed=True, overwrite=False, versioning=False, archive=False):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        file_name = f'{symbol}_data.csv'
        if processed:
            file_path = self.processed_csv_dir / file_name
        else:
            file_path = self.raw_csv_dir / file_name
        
        if versioning:
            version = 1
            base_name = file_name
            while file_path.exists():
                file_name = f'{base_name}_v{version}.csv'
                file_path = file_path.parent / file_name
                version += 1
        
        self.save_to_csv(data, file_name, overwrite)
        
        if archive:
            archive_dir = self.csv_dir / 'archive'
            self.utils.ensure_directory_exists(archive_dir)
            archive_path = archive_dir / file_name
            data.to_csv(archive_path, index=False)
        
        self.utils.logger.info(f"Saved data for {symbol} to {file_name}")

# Example usage when running the script independently
if __name__ == "__main__":
    store = DataStore()
    print("Available CSV files:", store.list_csv_files())
