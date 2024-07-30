import os
import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.regularizers import l1_l2

# Adjust the Python path dynamically for independent execution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
utilities_path = project_root / 'Scripts' / 'Utilities'
models_path = project_root / 'Scripts' / 'ModelTraining' / 'model_training' / 'models'
logs_path = project_root / 'logs'

sys.path.append(str(project_root))
sys.path.append(str(utilities_path))
sys.path.append(str(models_path))

# Import the configuration and utility functions
from config_handling import ConfigManager

# Create logs directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Set up logging to file and console
log_file_path = logs_path / 'model_training.log'
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[file_handler, console_handler])

logger = logging.getLogger("Model_Training")

# Import model trainers and utilities
from arima_model import ARIMAModelTrainer
from linear_regression import LinearRegressionModel
from lstm import LSTMModelTrainer, LSTMModelConfig
from neural_network import NeuralNetworkTrainer, NNModelConfig
from random_forest import RandomForestModel
from data_store import DataStore
from DataHandler import DataHandler

# Import the basic LSTM model
from basiclstm import basicLSTMModelTrainer, basicLSTMModelConfig

def handle_missing_values(data):
    """Handle missing values in the data."""
    logger.info("Handling missing values...")
    numeric_data = data.select_dtypes(include=[np.number])
    logger.debug(f"Numeric columns: {numeric_data.columns.tolist()}")
    imputer = SimpleImputer(strategy='mean')
    data[numeric_data.columns] = imputer.fit_transform(numeric_data)
    logger.info("Missing values handled")
    return data

def train_linear_regression(X_train, y_train, X_val, y_val):
    """Train a Linear Regression model."""
    logger.info("Training Linear Regression model...")
    lr_model = LinearRegressionModel(logger)
    lr_model.train(X_train, y_train, X_val, y_val)
    logger.info("Linear Regression training complete")

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Train an LSTM model."""
    logger.info("Training LSTM model...")
    lstm_trainer = LSTMModelTrainer(logger)
    
    # Define LSTM model parameters
    lstm_params = {
        'layers': [
            {'type': 'bidirectional_lstm', 'units': 100, 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
            {'type': 'attention'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 20, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)}
        ],
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    }

    # Initialize and compile the LSTM model with parameters
    model_config = LSTMModelConfig.lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), params=lstm_params)
    lstm_trainer.train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=50)
    logger.info("LSTM model training complete")

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train a Neural Network model."""
    logger.info("Training Neural Network model...")
    model_config = NNModelConfig.dense_model()
    nn_trainer = NeuralNetworkTrainer(model_config, epochs=50)
    nn_trainer.train(X_train, y_train, X_val)
    logger.info("Neural Network training complete")

def train_random_forest(X, y):
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    rf_model = RandomForestModel(logger=logger)
    rf_model.train(X, y, random_state=42)
    logger.info("Random Forest training complete")

def train_arima_model(close_prices):
    """Train an ARIMA model."""
    logger.info("Training ARIMA model...")
    arima_trainer = ARIMAModelTrainer(close_prices)
    arima_trainer.train()
    logger.info("ARIMA model training complete")

def train_basic_lstm_model(X_train, y_train, X_val, y_val):
    """Train a basic LSTM model."""
    logger.info("Training basic LSTM model...")
    basic_lstm_trainer = basicLSTMModelTrainer(logger)
    model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), params={})
    basic_lstm_trainer.train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=50)
    logger.info("Basic LSTM model training complete")

def main():
    logger.info("Starting model training script")

    # Initialize ConfigManager
    config_manager = ConfigManager()
    loading_path = config_manager.get('loading_path')
    api_key = config_manager.get('api_key')
    base_url = config_manager.get('base_url')
    timeout = config_manager.get('timeout')
    db_name = config_manager.get('db_name')
    db_user = config_manager.get('db_user')
    print(f"ConfigManager loaded values: Loading Path={loading_path}, API Key={api_key}, Base URL={base_url}, Timeout={timeout}, DB Name={db_name}, DB User={db_user}")

    # Initialize DataStore
    data_store = DataStore()

    # Initialize DataHandler
    data_handler = DataHandler(data_store=data_store)

    # List available CSV files
    available_files = data_store.list_csv_files()
    if not available_files:
        logger.error("No CSV files available in the data directory.")
        return

    file_name = input(f"Enter the CSV file name to load data from (one of {available_files}): ").strip()

    if file_name not in available_files:
        logger.error(f"File {file_name} not found in the data directory.")
        return

    # Load and preprocess data
    data = data_handler.load_data(f'C:/TheTradingRobotPlug/data/alpha_vantage/{file_name}')

    if data is not None:
        # Handle missing values before sequence creation
        data = handle_missing_values(data)

        # Split data into training, validation, and test sets
        logger.info("Splitting data into training, validation, and test sets...")
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)
        test_size = len(data) - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        logger.info(f"Data split - train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}")

        # Extract features and target
        non_numeric_columns = ['close', 'date', 'symbol']
        X_train = train_data.drop(columns=non_numeric_columns)  # Exclude non-numeric columns
        y_train = train_data['close']
        X_val = val_data.drop(columns=non_numeric_columns)  # Exclude non-numeric columns
        y_val = val_data['close']

        logger.debug(f"X_train columns after dropping non-numeric: {X_train.columns.tolist()}")
        logger.debug(f"X_val columns after dropping non-numeric: {X_val.columns.tolist()}")

        # Convert to numpy arrays
        X_train = X_train.values.astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        X_val = X_val.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)

        logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.debug(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        model_types = input("Enter the types of models to train (1: Linear Regression, 2: LSTM, 3: Neural Network, 4: Random Forest, 5: ARIMA, 6: Basic LSTM) separated by commas: ").strip().split(',')

        with ThreadPoolExecutor() as executor:
            futures = []
            try:
                for model_type in model_types:
                    if model_type.strip() == '1':
                        futures.append(executor.submit(train_linear_regression, X_train, y_train, X_val, y_val))
                    elif model_type.strip() == '2':
                        # Use LSTMModelTrainer's create_sequences method
                        lstm_trainer = LSTMModelTrainer(logger)
                        X_train_seq, y_train_seq = lstm_trainer.create_sequences(X_train, y_train, time_steps=10)
                        X_val_seq, y_val_seq = lstm_trainer.create_sequences(X_val, y_val, time_steps=10)
                        
                        # Reshape the data to (samples, time_steps, features)
                        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
                        X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], X_val_seq.shape[2]))

                        futures.append(executor.submit(train_lstm_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq))
                    elif model_type.strip() == '3':
                        futures.append(executor.submit(train_neural_network, X_train, y_train, X_val, y_val))
                    elif model_type.strip() == '4':
                        futures.append(executor.submit(train_random_forest, X_train, y_train))
                    elif model_type.strip() == '5':
                        futures.append(executor.submit(train_arima_model, train_data['close']))
                    elif model_type.strip() == '6':
                        # Use basicLSTMModelTrainer's create_sequences method
                        basic_lstm_trainer = basicLSTMModelTrainer(logger)
                        X_train_seq, y_train_seq = basic_lstm_trainer.create_sequences(X_train, y_train, time_steps=10)
                        X_val_seq, y_val_seq = basic_lstm_trainer.create_sequences(X_val, y_val, time_steps=10)
                        
                        # Reshape the data to (samples, time_steps, features)
                        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
                        X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], X_val_seq.shape[2]))

                        futures.append(executor.submit(train_basic_lstm_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq))
                    else:
                        logger.error(f"Invalid model type: {model_type}")

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"An error occurred: {str(e)}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"An error occurred while scheduling the model training: {str(e)}")
                logger.error(traceback.format_exc())

    logger.info("Model training script completed")

if __name__ == "__main__":
    main()
