# C:\TheTradingRobotPlug\Scripts\Utilities\config_handling.py

import os
import logging
import numpy as np
from sklearn.metrics import mean_squared_error

# Function to safely get an environment variable with a default
def get_env_value(key, default=None):
    return os.getenv(key, default)

# Safely get the environment variables
loading_path = get_env_value('LOADING_PATH', 'default/loading/path')
api_key = get_env_value('API_KEY', 'default_api_key')
base_url = get_env_value('BASE_URL', 'https://api.example.com')
timeout = int(get_env_value('TIMEOUT', 30))
db_name = get_env_value('DB_NAME', 'default_db')
db_user = get_env_value('DB_USER', 'default_user')

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Print statements for debugging
print(f"Loading Path: {loading_path}")
print(f"API Key: {api_key}")
print(f"Base URL: {base_url}")
print(f"Timeout: {timeout}")
print(f"Database Name: {db_name}")
print(f"Database User: {db_user}")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example usage of the logger
logger.info(f"Configuration loaded successfully: API Key={api_key}, Base URL={base_url}, Timeout={timeout}, DB Name={db_name}, DB User={db_user}")

class ConfigManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = {
            'loading_path': loading_path,
            'api_key': api_key,
            'base_url': base_url,
            'timeout': timeout,
            'db_name': db_name,
            'db_user': db_user,
        }

    def get(self, key):
        value = self.config.get(key)
        self.logger.debug(f"Retrieving {key}: {value}")
        return value

# Function to calculate RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Example usage of ConfigManager and RMSE function
if __name__ == "__main__":
    config_manager = ConfigManager()
    loading_path = config_manager.get('loading_path')
    api_key = config_manager.get('api_key')
    base_url = config_manager.get('base_url')
    timeout = config_manager.get('timeout')
    db_name = config_manager.get('db_name')
    db_user = config_manager.get('db_user')
    print(f"ConfigManager loaded values: Loading Path={loading_path}, API Key={api_key}, Base URL={base_url}, Timeout={timeout}, DB Name={db_name}, DB User={db_user}")

    # Example RMSE calculation (replace y_true and y_pred with actual data)
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"Calculated RMSE: {rmse}")
