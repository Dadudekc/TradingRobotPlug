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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

class AdvancedLSTMModelTrainer:
    def __init__(self, logger_handler, config_manager, model_save_path=None, scaler_save_path=None):
        self.logger = logger_handler
        self.config_manager = config_manager
        self.model_save_path = Path(model_save_path or self.config_manager.get('model_save_path', 'best_model.keras'))
        self.scaler_save_path = Path(scaler_save_path or self.config_manager.get('scaler_save_path', 'scaler.pkl'))

    def preprocess_data(self, X_train, X_val):
        self.logger.log(level=logging.INFO, msg="Preprocessing data...")
        self.logger.log(level=logging.DEBUG, msg=f"Initial X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        joblib.dump(scaler, self.scaler_save_path)

        self.logger.log(level=logging.DEBUG, msg=f"Scaled X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}")
        return X_train_scaled, X_val_scaled

    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, callbacks=None):
        self.logger.log(level=logging.INFO, msg="Starting LSTM model training...")
        
        try:
            # Reshape X_train and X_val to match LSTM input requirements
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], -1))

            self.logger.log(level=logging.INFO, msg=f"X_train_scaled shape: {X_train_reshaped.shape}")
            self.logger.log(level=logging.INFO, msg=f"y_train shape: {y_train.shape}")
            self.logger.log(level=logging.INFO, msg=f"X_val_scaled shape: {X_val_reshaped.shape}")
            self.logger.log(level=logging.INFO, msg=f"y_val shape: {y_val.shape}")

            model = self.build_lstm_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

            if callbacks is None:
                callbacks = []

            model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32,
                    callbacks=callbacks)

            y_pred_val = model.predict(X_val_reshaped).flatten()

            self.logger.log(level=logging.INFO, msg=f"Predicted y_val shape: {y_pred_val.shape}")
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = root_mean_squared_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.log(level=logging.INFO, msg=f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            model.save(self.model_save_path)
            return model
        except Exception as e:
            self.logger.log(level=logging.ERROR, msg=f"Error occurred during LSTM model training: {e}")
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
            self.logger.error(f"Error occurred during model evaluation: {e}")

    def create_sequences(self, data, target, time_steps=10):
        xs, ys = [], []
        for i in range(len(data) - time_steps):
            x = data[i:i + time_steps]
            y = target[i + time_steps]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

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

def main():
    # Manually handle the default value for data_file_path
    data_file_path = config_manager.get('data_file_path')
    if data_file_path is None:
        data_file_path = project_root / 'data' / 'alpha_vantage' / 'tsla_data.csv'

    logger_handler = LoggerHandler(logger=logger)
    data_loader = DataLoader(logger_handler)
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load data using DataLoader
    data = data_loader.load_data(data_file_path)

    if data is not None:
        # Expecting four values to be returned
        X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(data)

        if X_train is not None and y_train is not None:
            time_steps = 10

            logger_handler.log(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")

            trainer = AdvancedLSTMModelTrainer(logger_handler, config_manager)
            try:
                logger_handler.log(f"Creating sequences with time_steps: {time_steps}")
                X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

                logger_handler.log(f"X_train_seq shape: {X_train_seq.shape}")
                logger_handler.log(f"y_train_seq shape: {y_train_seq.shape}")
                logger_handler.log(f"X_val_seq shape: {X_val_seq.shape}")
                logger_handler.log(f"y_val_seq shape: {y_val_seq.shape}")

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                trained_model = trainer.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50,
                    callbacks=[early_stopping, reduce_lr]
                )

                if trained_model:
                    trainer.evaluate_model(X_val_seq, y_val_seq)
            except ValueError as e:
                logger_handler.log(f"ValueError in creating sequences: {e}", "ERROR")
            except KeyError as e:
                logger_handler.log(f"KeyError encountered: {e}", "ERROR")
        else:
            logger_handler.log("Data preprocessing failed.", "ERROR")
    else:
        logger_handler.log("Data loading failed.", "ERROR")

if __name__ == "__main__":
    main()
