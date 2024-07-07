import os
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add project root to the Python path for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.data_fetch_utils import DataFetchUtils
from Scripts.Utilities.config_handling import ConfigManager

class DataStore:
    def __init__(self, csv_dir='C:/TheTradingRobotPlug/data/csv', db_path='C:/TheTradingRobotPlug/data/trading_data.db'):
        self.csv_dir = csv_dir
        self.db_path = db_path
        self.data = {}
        self.utils = DataFetchUtils("C:/TheTradingRobotPlug/logs/data_store.log")
        self.utils.ensure_directory_exists(self.csv_dir)
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
        file_path = os.path.join(self.csv_dir, file_name)
        self.utils.save_data_to_csv(df, file_path, overwrite)

    def save_to_sql(self, df, table_name, if_exists='replace'):
        self.utils.save_data_to_sql(df, table_name, self.db_path, if_exists)

    def fetch_from_csv(self, file_name):
        file_path = os.path.join(self.csv_dir, file_name)
        return pd.read_csv(file_path)

    def fetch_from_sql(self, table_name):
        return self.utils.fetch_data_from_sql(table_name, self.db_path)

    def list_csv_files(self):
        return [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]

# Example usage when running the script independently
if __name__ == "__main__":
    store = DataStore()
    print("Available CSV files:", store.list_csv_files())
