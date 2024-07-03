# C:\TheTradingRobotPlug\Scripts\Utilities\data_fetch_utils.py

import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from config_handling import load_config, load_paths

def setup_logger(logger_name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    logs_dir = os.path.dirname(log_file)
    os.makedirs(logs_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def ensure_directory_exists(directory: str):
    os.makedirs(directory, exist_ok=True)

def save_data_to_csv(data, file_path, overwrite=False):
    if not overwrite and os.path.exists(file_path):
        logger = logging.getLogger("DataFetchUtility")
        logger.warning(f"File {file_path} already exists and overwrite is set to False.")
        return
    
    try:
        data.to_csv(file_path, index=False)
        logger = logging.getLogger("DataFetchUtility")
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger = logging.getLogger("DataFetchUtility")
        logger.error(f"Error saving data to CSV: {e}")

def save_data_to_sql(df, table_name, db_path='data/trading_data.db', if_exists='replace'):
    ensure_directory_exists(os.path.dirname(db_path))
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        logger = logging.getLogger("DataFetchUtility")
        logger.info(f"Data saved to SQL table {table_name} in {db_path}")
    except Exception as e:
        logger = logging.getLogger("DataFetchUtility")
        logger.error(f"Error saving data to SQL: {e}")

def fetch_data_from_sql(table_name, db_path='data/trading_data.db'):
    engine = create_engine(f'sqlite:///{db_path}')
    try:
        df = pd.read_sql_table(table_name, engine)
        return df
    except Exception as e:
        logger = logging.getLogger("DataFetchUtility")
        logger.error(f"Error fetching data from SQL: {e}")
        return None
