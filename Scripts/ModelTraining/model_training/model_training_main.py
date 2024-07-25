import os
import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
import traceback
import numpy as np

# Adjust the Python path dynamically for independent execution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
utilities_path = project_root / 'Scripts' / 'Utilities'
models_path = project_root / 'Scripts' / 'ModelTraining' / 'model_training' / 'models'
logs_path = project_root / 'logs'

sys.path.append(str(project_root))
sys.path.append(str(utilities_path))
sys.path.append(str(models_path))

# Create logs directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Set up logging to file
log_file_path = logs_path / 'model_training.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

logger = logging.getLogger("Model_Training")

# Import model trainers and utilities
from arima_model import ARIMAModelTrainer
from linear_regression import LinearRegressionModel
from lstm import LSTMModelTrainer, LSTMModelConfig
from neural_network import NeuralNetworkTrainer, ModelConfig
from random_forest import RandomForestModel
from data_store import DataStore
from DataHandler import DataHandler

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

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
    time_steps = 10  # Define the number of time steps for the LSTM input
    X_train_seq = create_sequences(X_train, time_steps)
    y_train_seq = y_train[time_steps:]

    X_val_seq = create_sequences(X_val, time_steps)
    y_val_seq = y_val[time_steps:]

    logger.debug(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    logger.debug(f"X_val_seq shape: {X_val_seq.shape}, y_val_seq shape: {y_val_seq.shape}")

    # Additional logging to debug the shapes
    logger.debug(f"X_train_seq sample: {X_train_seq[:2]}")
    logger.debug(f"y_train_seq sample: {y_train_seq[:2]}")
    logger.debug(f"X_val_seq sample: {X_val_seq[:2]}")
    logger.debug(f"y_val_seq sample: {y_val_seq[:2]}")

    if X_train_seq.shape[0] != y_train_seq.shape[0] or X_val_seq.shape[0] != y_val_seq.shape[0]:
        raise ValueError(f"Shape mismatch between X and y sequences: X_train_seq {X_train_seq.shape}, y_train_seq {y_train_seq.shape}, X_val_seq {X_val_seq.shape}, y_val_seq {y_val_seq.shape}")

    model_config = LSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_trainer = LSTMModelTrainer(logger)

    lstm_trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config)
    logger.info("LSTM training complete")

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train a Neural Network model."""
    logger.info("Training Neural Network model...")
    model_config = ModelConfig.dense_model()
    nn_trainer = NeuralNetworkTrainer(model_config, logger, epochs=50)
    nn_trainer.train(X_train, y_train, X_val, y_val)
    logger.info("Neural Network training complete")

def train_random_forest(X, y):
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    rf_model = RandomForestModel(logger=logger)
    rf_model.train(X, y, random_state=42)
    logger.info("Random Forest training complete")

def main():
    logger.info("Starting model training script")

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

        model_type = input("Enter the type of model to train (1: Linear Regression, 2: LSTM, 3: Neural Network, 4: Random Forest): ").strip()

        try:
            if model_type == '1':
                train_linear_regression(X_train, y_train, X_val, y_val)
            elif model_type == '2':
                train_lstm_model(X_train, y_train, X_val, y_val)
            elif model_type == '3':
                train_neural_network(X_train, y_train, X_val, y_val)
            elif model_type == '4':
                train_random_forest(X_train, y_train)
            else:
                logger.error(f"Invalid model type: {model_type}")
        except Exception as e:
            logger.error(f"An error occurred while training the model: {str(e)}")
            logger.error(traceback.format_exc())

    logger.info("Model training script completed")

if __name__ == '__main__':
    main()
