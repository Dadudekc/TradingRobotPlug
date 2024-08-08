import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit  # For cross-validation

# Adjust import path based on your project structure
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Assuming project root is two levels up

# Add the correct directories to sys.path
model_training_dir = project_root / 'ModelTraining' / 'model_training'
utilities_dir = project_root / 'Utilities'

sys.path.append(str(model_training_dir))
sys.path.append(str(utilities_dir))

# Debug print to confirm the paths
print("DEBUG: Corrected Project root path:", project_root)
print("DEBUG: Adding ModelTraining directory to sys.path:", model_training_dir)
print("DEBUG: Adding Utilities directory to sys.path:", utilities_dir)

# Importing necessary utilities
from model_training_utils import (
    setup_logger, load_model_from_file, save_predictions, save_metadata, 
    validate_predictions, preprocess_data, detect_models, DataPreprocessor, 
    check_for_nan_inf, LoggerHandler, DataLoader
)

# Attempt to import additional modules
try:
    from config_handling import ConfigManager
    from data_store import DataStore
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Define classes and functions

class basicLSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape):
        if len(input_shape) != 2:
            raise ValueError("Input shape must be a tuple of (time_steps, features)")
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

class basicLSTMModelTrainer:
    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def create_sequences(data, target, time_steps=10):
        sequences = []
        targets = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            targets.append(target[i + time_steps])
        return np.array(sequences), np.array(targets)

    def train_lstm(self, X_train, y_train, X_val, y_val, model_config, epochs=50):
        self.logger.info("Initializing the LSTM model...")
        model = model_config
        
        self.logger.info("Starting model training...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
        self.logger.info("LSTM model training complete")
        
        return model, history

def prepare_data(data, target_column='close', time_steps=10):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    data = data.copy()
    
    # Select all numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    target = data[target_column].values
    sequences, targets = basicLSTMModelTrainer.create_sequences(scaled_data, target, time_steps)
    
    return sequences, targets, scaler

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.title('LSTM Model Predictions vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

def cross_validate_lstm(data, target_column='close', time_steps=10, n_splits=5, epochs=50):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    r2_scores = []
    
    for train_index, val_index in tscv.split(data):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = data[target_column].values[train_index], data[target_column].values[val_index]
        
        X_train_seq, y_train_seq = basicLSTMModelTrainer.create_sequences(X_train, y_train, time_steps)
        X_val_seq, y_val_seq = basicLSTMModelTrainer.create_sequences(X_val, y_val, time_steps)
        
        model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        trainer = basicLSTMModelTrainer(logger)
        model, _ = trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs)
        
        predictions = model.predict(X_val_seq)
        mse_scores.append(mean_squared_error(y_val_seq, predictions))
        r2_scores.append(r2_score(y_val_seq, predictions))
    
    print(f"Cross-Validation Mean Squared Error Scores: {mse_scores}")
    print(f"Cross-Validation R² Scores: {r2_scores}")
    print(f"Mean Cross-Validation MSE: {np.mean(mse_scores)}")
    print(f"Mean Cross-Validation R²: {np.mean(r2_scores)}")

def list_csv_files(directory):
    """List all CSV files in a directory."""
    print(f"DEBUG: Checking for CSV files in directory: {directory}")
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print("DEBUG: No CSV files found in the directory.")
        return None
    print("DEBUG: Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"DEBUG: {i}: {file}")
    return csv_files

def select_csv_file(directory):
    """Prompt the user to select a CSV file from a directory."""
    csv_files = list_csv_files(directory)
    if not csv_files:
        return None
    try:
        choice = int(input("Enter the number of the file to use: "))
        if 1 <= choice <= len(csv_files):
            return os.path.join(directory, csv_files[choice - 1])
        else:
            print("DEBUG: Invalid choice.")
            return None
    except ValueError:
        print("DEBUG: Please enter a valid number.")
        return None

if __name__ == "__main__":
    logger = setup_logger("LSTM_Training")
    logger_handler = LoggerHandler(logger=logger)
    config_manager = ConfigManager()
    data_loader = DataLoader(logger_handler)
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Correct the data directory path
    data_dir = Path("C:/TheTradingRobotPlug/data/alpha_vantage")
    print(f"DEBUG: Data directory path: {data_dir}")
    selected_file = select_csv_file(data_dir)
    if not selected_file:
        sys.exit("No valid file selected. Exiting.")
    
    # Load data
    data = data_loader.load_data(selected_file)
    if data is None:
        sys.exit("Data loading failed. Exiting.")

    # Data preprocessing
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, scaler = prepare_data(data)

    # Ensure data shape is correct
    logger.info(f"X_train_seq shape: {X_train_seq.shape}")
    logger.info(f"y_train_seq shape: {y_train_seq.shape}")
    logger.info(f"X_val_seq shape: {X_val_seq.shape}")
    logger.info(f"y_val_seq shape: {y_val_seq.shape}")
    
    # Convert to NumPy array and reshape the data to add the third dimension
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
    X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], X_val_seq.shape[2]))

    lstm_trainer = basicLSTMModelTrainer(logger)
    
    # Make sure the shape of X_train_seq is as expected
    try:
        model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    except (IndexError, ValueError) as e:
        logger.error(f"Error in model configuration: {e}")
        raise
    
    # Train the model
    model, history = lstm_trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50)
    
    # Make predictions
    predictions = model.predict(X_val_seq)
    
    # Rescale predictions and actual values
    y_val_seq_rescaled = scaler.inverse_transform(y_val_seq.reshape(-1, 1))
    predictions_rescaled = scaler.inverse_transform(predictions)
    
    # Plot predictions vs actual values (unscaled)
    plot_predictions(y_val_seq_rescaled, predictions_rescaled)
    
    # Evaluate model
    evaluate_model(y_val_seq_rescaled, predictions_rescaled)
    
    # Cross-validation
    cross_validate_lstm(data.values, target_column='close', time_steps=10, n_splits=5, epochs=50)
