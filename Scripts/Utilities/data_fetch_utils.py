import logging
import os
import pandas as pd
import sqlite3
from contextlib import closing
from sqlalchemy import create_engine

class DataFetchUtils:
    def __init__(self, log_file='logs/data_fetch_utils.log'):
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def ensure_directory_exists(self, directory):
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

# Example of how to use the DataFetchUtils class
if __name__ == "__main__":
    utils = DataFetchUtils()
    utils.ensure_directory_exists('data/csv')
    print("Logger and utility functions initialized.")
    utils.close_logger()
