# C:\TheTradingRobotPlug\Scripts\Utilities\data_fetch_utils.py

import logging
import os
import sys
import pandas as pd
import sqlite3
from contextlib import closing
from sqlalchemy import create_engine

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.config_handling import ConfigManager

class DataFetchUtils:
    def __init__(self, config_file='config.ini', log_file='logs/data_fetch_utils.log', log_text_widget=None):
        # Initialize ConfigManager
        defaults = {
            'LOGGING': {
                'log_level': 'DEBUG'
            },
            'DIRECTORIES': {
                'data_dir': 'data/csv'
            }
        }
        self.config_manager = ConfigManager(config_file=config_file, defaults=defaults)

        # Set up logger
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        log_level = self.config_manager.get('LOGGING', 'log_level', 'DEBUG')
        self.logger.setLevel(getattr(logging, log_level))

        self.log_text_widget = log_text_widget

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            self.logger.log(getattr(logging, level.upper()), message)

    def ensure_directory_exists(self, directory):
        if directory is None:
            self.logger.error("Directory path is None. Please check the configuration.")
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.logger.info(f"Ensured directory exists: {directory}")

    def save_data_to_csv(self, df, file_path, overwrite=False):
        if not overwrite and os.path.exists(file_path):
            self.logger.warning(f"File {file_path} already exists and overwrite is set to False.")
            return
        df.to_csv(file_path, index=False)
        self.logger.info(f"Data saved to CSV at {file_path}")

    def save_data_to_sql(self, df, table_name, db_path, if_exists='replace'):
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            self.logger.info(f"Data saved to SQL table {table_name} in database {db_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to SQL table {table_name} in database {db_path}: {e}")
            raise

    def fetch_data_from_sql(self, table_name, db_path):
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            self.logger.info(f"Data fetched from SQL table {table_name} in database {db_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from SQL table {table_name} in database {db_path}: {e}")
            raise

    def setup_logger(self, name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.hasHandlers():
            logger.addHandler(handler)

        return logger

    def close_logger(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def validate_and_preprocess_data(self, file_path: str, required_columns: list, target_column: str = 'close', date_column: str = 'date', lag_sizes: list = [1, 2, 3, 5, 10], window_sizes: list = [5, 10, 20], scaler_type: str = 'StandardScaler'):
        """
        Loads, validates, and preprocesses the data.
        """
        try:
            df = self.data_handler.load_data(file_path)
            DataValidation.validate_dataframe(df, required_columns)
            if DataValidation.check_for_nulls(df):
                df = DataCleaning.fill_missing_values(df, strategy='mean')
            X_train, X_val, y_train, y_val = self.data_handler.preprocess_data(df, target_column, date_column, lag_sizes, window_sizes, scaler_type)
            self.log("Data validation and preprocessing completed.")
            return X_train, X_val, y_train, y_val
        except Exception as e:
            self.log(f"Error in data validation and preprocessing: {e}", "ERROR")
            return None, None, None, None

    def save_preprocessed_data(self, X_train, X_val, y_train, y_val, symbol: str):
        """
        Saves the preprocessed data.
        """
        try:
            self.data_store.save_data(pd.concat([X_train, y_train], axis=1), f"{symbol}_train")
            self.data_store.save_data(pd.concat([X_val, y_val], axis=1), f"{symbol}_val")
            self.log("Preprocessed data saved successfully.")
        except Exception as e:
            self.log(f"Error saving preprocessed data: {e}", "ERROR")

# Example of how to use the DataFetchUtils class
if __name__ == "__main__":
    config_file = 'config.ini'
    utils = DataFetchUtils(config_file=config_file)
    data_dir = utils.config_manager.get('DIRECTORIES', 'data_dir')
    print(f"Configured data directory: {data_dir}")  # Debugging print
    utils.ensure_directory_exists(data_dir)
    print("Logger and utility functions initialized.")
    utils.close_logger()
