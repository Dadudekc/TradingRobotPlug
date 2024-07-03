# C:\TheTradingRobotPlug\Scripts\Utilities\data_store.py

import os
import pandas as pd
from data_fetch_utils import setup_logger, ensure_directory_exists, save_data_to_csv, save_data_to_sql, fetch_data_from_sql

class DataStore:
    def __init__(self, csv_dir='data/csv', db_path='data/trading_data.db'):
        self.csv_dir = csv_dir
        self.db_path = db_path
        self.logger = setup_logger("DataStore", "logs/data_store.log")
        ensure_directory_exists(self.csv_dir)

    def save_to_csv(self, df, file_name, overwrite=False):
        file_path = os.path.join(self.csv_dir, file_name)
        save_data_to_csv(df, file_path, overwrite)

    def save_to_sql(self, df, table_name, if_exists='replace'):
        save_data_to_sql(df, table_name, self.db_path, if_exists)

    def fetch_from_csv(self, file_name):
        file_path = os.path.join(self.csv_dir, file_name)
        return pd.read_csv(file_path)

    def fetch_from_sql(self, table_name):
        return fetch_data_from_sql(table_name, self.db_path)

    def list_csv_files(self):
        return [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
