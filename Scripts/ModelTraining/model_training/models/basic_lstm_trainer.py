import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), 
                            callbacks=[early_stopping, reduce_lr])
        self.logger.info("LSTM model training complete")
        
        return model, history

def prepare_data(data, target_column='close', time_steps=10, split_ratio=0.8):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Check for NaN or infinite values and handle them
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Exclude non-numeric columns (e.g., date columns) from the feature set
    features = data.drop(columns=[target_column])
    numeric_features = features.select_dtypes(include=[np.number])
    
    # Scale numeric features and the target column separately
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(numeric_features)
    scaled_target = target_scaler.fit_transform(data[[target_column]])

    # Combine scaled features and scaled target
    scaled_data = np.hstack((scaled_features, scaled_target))

    # Create sequences
    sequences, targets = basicLSTMModelTrainer.create_sequences(scaled_data, scaled_target.flatten(), time_steps)
    
    # Split the data into training and validation sets
    split_index = int(len(sequences) * split_ratio)
    X_train_seq = sequences[:split_index]
    y_train_seq = targets[:split_index]
    X_val_seq = sequences[split_index:]
    y_val_seq = targets[split_index:]
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, target_scaler

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
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")

def cross_validate_lstm(data, target_column='close', time_steps=10, n_splits=5, epochs=50):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    mae_scores = []
    r2_scores = []
    
    numeric_data = data.drop(columns=['date', 'symbol']).select_dtypes(include=[np.number]).values
    target_column_index = data.columns.get_loc(target_column)
    target_data = numeric_data[:, target_column_index]

    for i, (train_index, val_index) in enumerate(tscv.split(numeric_data)):
        print(f"\n--- Cross-Validation Fold {i + 1} ---")

        X_train, X_val = numeric_data[train_index], numeric_data[val_index]
        y_train, y_val = target_data[train_index], target_data[val_index]

        print("\nSample of X_train:")
        print(pd.DataFrame(X_train).head(10))
        
        print("\nSample of y_train:")
        print(pd.DataFrame(y_train).head(10))

        if np.isnan(X_train).any() or np.isnan(y_train).any():
            logger.error("Training data contains NaN values.")
            print("Training data contains NaN values.")
            print("NaN found in X_train:", np.isnan(X_train).any())
            print("NaN found in y_train:", np.isnan(y_train).any())
            return
        if np.isnan(X_val).any() or np.isnan(y_val).any():
            logger.error("Validation data contains NaN values.")
            print("Validation data contains NaN values.")
            print("NaN found in X_val:", np.isnan(X_val).any())
            print("NaN found in y_val:", np.isnan(y_val).any())
            return
        
        X_train_seq, y_train_seq = basicLSTMModelTrainer.create_sequences(X_train, y_train, time_steps)
        X_val_seq, y_val_seq = basicLSTMModelTrainer.create_sequences(X_val, y_val, time_steps)
        
        trainer = basicLSTMModelTrainer(logger)
        model, _ = trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config=basicLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])), epochs=epochs)
        
        predictions = model.predict(X_val_seq)
        
        if np.isnan(predictions).any():
            logger.error("Model predictions contain NaN values.")
            print("Model predictions contain NaN values.")
            return

        mse_scores.append(mean_squared_error(y_val_seq, predictions))
        mae_scores.append(mean_absolute_error(y_val_seq, predictions))
        r2_scores.append(r2_score(y_val_seq, predictions))
    
    print(f"Cross-Validation Mean Squared Error Scores: {mse_scores}")
    print(f"Cross-Validation Mean Absolute Error Scores: {mae_scores}")
    print(f"Cross-Validation R² Scores: {r2_scores}")
    print(f"Mean Cross-Validation MSE: {np.mean(mse_scores)}")
    print(f"Mean Cross-Validation MAE: {np.mean(mae_scores)}")
    print(f"Mean Cross-Validation R²: {np.mean(r2_scores)}")

def list_csv_files(directory):
    """List all CSV files in a directory."""
    print(f"DEBUG: Checking for CSV files in directory: {directory}")
    csv_files
