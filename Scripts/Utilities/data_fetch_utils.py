import os
import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Dict

# Conditional import based on script execution context
if __name__ == "__main__":
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))
    from Scripts.Utilities.config_handling import ConfigManager
else:
    from .config_handling import ConfigManager

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Function to set up a logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    return logger

def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def save_data_to_csv(df, file_path: str, overwrite: bool = False) -> None:
    """Save DataFrame to CSV."""
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists.")
    df.to_csv(file_path, index=False)

def save_data_to_sql(df, table_name: str, db_path: str, if_exists: str = 'replace') -> None:
    """Save DataFrame to SQL database."""
    from sqlalchemy import create_engine
    engine = create_engine(f'sqlite:///{db_path}')
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)

def fetch_data_from_sql(table_name: str, db_path: str):
    """Fetch data from SQL database."""
    from sqlalchemy import create_engine
    import pandas as pd
    
    engine = create_engine(f'sqlite:///{db_path}')
    return pd.read_sql(table_name, con=engine)

if __name__ == "__main__":
    # Example usage when running the script independently
    config_manager = ConfigManager()  # Initialize ConfigManager
    config = config_manager.load_config()  # Load configuration
    paths = config_manager.get_paths()  # Get paths from configuration
    user_settings = config_manager.get_user_settings()  # Get user settings
    
    print("Configuration loaded:", config)
    print("Paths loaded:", paths)
    print("User settings loaded:", user_settings)
