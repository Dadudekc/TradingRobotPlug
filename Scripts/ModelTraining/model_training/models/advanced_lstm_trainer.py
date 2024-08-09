# File: lstm_model_trainer.py
# Location: Scripts/Training/
# Description: Script for training LSTM models on time series data with hyperparameter optimization and logging.


import os
import sys
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, BatchNormalization
import matplotlib.pyplot as plt
import optuna

# Set up project root and add 'Utilities' to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # Adjust based on your project structure
utilities_dir = project_root / 'Scripts' / 'Utilities'

# Add the Utilities directory to sys.path
if utilities_dir.exists() and str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

# Set up relative paths for resources and logs
resources_path = project_root / 'resources'
log_path = project_root / 'logs'

# Ensure the directories exist
resources_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
log_file = log_path / 'lstm_model_trainer.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger('AdvancedLSTMModelTrainer')

# Import ConfigManager and utilities
try:
    from config_handling import ConfigManager, root_mean_squared_error, get_env_value
    from model_training_utils import LoggerHandler, DataLoader, DataPreprocessor, check_for_nan_inf
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

# Initialize ConfigManager and LoggerHandler
config_manager = ConfigManager()
logger_handler = LoggerHandler(logger=logger)

# Setup log directory using ConfigManager
log_dir = Path(config_manager.get('log_dir', log_path))
log_dir.mkdir(parents=True, exist_ok=True)

class RefinedLSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0), loss='mean_squared_error')
        return model

class AdvancedLSTMModelTrainer:
    def __init__(self, logger_handler, config_manager, model_save_path=None, scaler_save_path=None):
        self.logger = logger_handler
        self.config_manager = config_manager
        self.model_save_path = Path(model_save_path or self.config_manager.get('model_save_path', 'best_model.keras'))
        self.scaler_save_path = Path(scaler_save_path or self.config_manager.get('scaler_save_path', 'scaler.pkl'))

    def preprocess_data(self, X_train, X_val):
        self.logger.log("Preprocessing data...", level=logging.INFO)
        self.logger.log(f"Initial X_train shape: {X_train.shape}, X_val shape: {X_val.shape}", level=logging.DEBUG)

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        joblib.dump(scaler, self.scaler_save_path)

        self.logger.log(f"Scaled X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}", level=logging.DEBUG)
        return X_train_scaled, X_val_scaled

    def build_lstm_model(self, input_shape):
        return RefinedLSTMModelConfig.lstm_model(input_shape)

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, callbacks=None):
        self.logger.log("Starting LSTM model training...", level=logging.INFO)
        
        try:
            # Reshape X_train and X_val to match LSTM input requirements
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], -1))

            self.logger.log(f"X_train_scaled shape: {X_train_reshaped.shape}", level=logging.INFO)
            self.logger.log(f"y_train shape: {y_train.shape}", level=logging.INFO)
            self.logger.log(f"X_val_scaled shape: {X_val_reshaped.shape}", level=logging.INFO)
            self.logger.log(f"y_val shape: {y_val.shape}", level=logging.INFO)

            model = self.build_lstm_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

            if callbacks is None:
                callbacks = []

            # Fit the model and capture the history
            history = model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32,
                                callbacks=callbacks)

            y_pred_val = model.predict(X_val_reshaped).flatten()

            self.logger.log(f"Predicted y_val shape: {y_pred_val.shape}", level=logging.INFO)
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = root_mean_squared_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.log(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", level=logging.INFO)

            model.save(self.model_save_path)
            return model, history
        except Exception as e:
            self.logger.log(f"Error occurred during LSTM model training: {e}", level=logging.ERROR)
            raise

    def evaluate_model(self, X_test, y_test):
        try:
            model = tf.keras.models.load_model(self.model_save_path)
            scaler = joblib.load(self.scaler_save_path)

            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            y_pred_test = model.predict(X_test_scaled).flatten()

            mse = mean_squared_error(y_test, y_pred_test)
            rmse = root_mean_squared_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)

            self.logger.log(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", level="INFO")
        except Exception as e:
            self.logger.log(f"Error occurred during model evaluation: {e}", level="ERROR")

    @staticmethod
    def create_sequences(data, target, time_steps=10):
        sequences = []
        targets = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            targets.append(target[i + time_steps])
        return np.array(sequences), np.array(targets)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        self.logger.log(f"X_train shape: {X_train.shape}")
        self.logger.log(f"X_val shape: {X_val.shape}")

        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model = self.train_lstm(X_train, y_train, X_val, y_val, epochs=50)

        y_pred_val = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        return mse

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        self.logger.log(f"Best hyperparameters: {study.best_params}")
        return study.best_params

def prepare_data(data, target_column='close', time_steps=10):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    data = data.copy()
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Handle NaN and infinite values
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    target = scaled_data[:, data.columns.get_loc(target_column)]
    sequences, targets = AdvancedLSTMModelTrainer.create_sequences(scaled_data, target, time_steps)
    
    return sequences, targets, scaler

def plot_predictions(y_true_scaled, y_pred_scaled, scaler, target_column):
    # Inverse transform to get back to original scale
    y_true = scaler.inverse_transform(np.concatenate([np.zeros((len(y_true_scaled), scaler.n_features_in_ - 1)), y_true_scaled.reshape(-1, 1)], axis=1))[:, -1]
    y_pred = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1)), y_pred_scaled.reshape(-1, 1)], axis=1))[:, -1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.title('LSTM Model Predictions vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

def main():
    # Manually handle the default value for data_file_path
    data_file_path = config_manager.get('data_file_path')
    if data_file_path is None:
        data_file_path = project_root / 'data' / 'alpha_vantage' / 'tsla_data.csv'

    logger_handler = LoggerHandler(logger=logger)
    data_loader = DataLoader(logger_handler)
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load data using DataLoader
    data = pd.read_csv(data_file_path)

    if data is not None:
        # Prepare the data
        X_train_seq, y_train_seq, scaler = prepare_data(data, 'close', time_steps=10)
        X_val_seq, y_val_seq, _ = prepare_data(data, 'close', time_steps=10)

        # Ensure data shape is correct
        logger_handler.log(f"X_train_seq shape: {X_train_seq.shape}")
        logger_handler.log(f"y_train_seq shape: {y_train_seq.shape}")
        logger_handler.log(f"X_val_seq shape: {X_val_seq.shape}")
        logger_handler.log(f"y_val_seq shape: {y_val_seq.shape}")

        trainer = AdvancedLSTMModelTrainer(logger_handler, config_manager)
        
        # Make sure the shape of X_train_seq is as expected
        try:
            model_config = RefinedLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        except (IndexError, ValueError) as e:
            logger.error(f"Error in model configuration: {e}")
            raise
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        model, history = trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, callbacks=[early_stopping, reduce_lr])

        # Make predictions
        predictions = model.predict(X_val_seq)
        
        # Plot predictions vs actual values (unscaled) and print evaluation metrics
        plot_predictions(y_val_seq, predictions, scaler, 'close')
    else:
        logger_handler.log("Data loading failed.", level="ERROR")
