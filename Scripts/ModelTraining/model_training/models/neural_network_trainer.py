# File: neural_network_trainer.py
# Location: Scripts/ModelTraining/model_training/models
# Description: Training and managing neural network models for stock prediction

import os
import sys
import logging
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Ensure TensorFlow and Keras components are imported
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

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

# Logging configuration
log_file = project_root / 'logs' / 'application.log'
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Enable Mixed Precision Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class NeuralNetworkTrainer:
    def __init__(self, model_config, epochs=100, pretrained_model_path=None, ticker=None):
        self.model_config = model_config
        self.epochs = epochs
        self.pretrained_model_path = pretrained_model_path
        self.ticker = ticker
        self.model = None
        self.strategy = tf.distribute.MirroredStrategy()

    def scheduler(self, epoch, lr):
        new_lr = lr if epoch < 10 else lr * tf.math.exp(-0.1)
        return float(new_lr)

    def build_model(self, input_shape):
        layer_mapping = {
            'dense': Dense,
            'batch_norm': BatchNormalization,
            'dropout': Dropout,
            'lstm': LSTM,
            'gru': GRU
        }

        with self.strategy.scope():
            if self.pretrained_model_path:
                self.model = load_model_from_file(self.pretrained_model_path)
                for layer in self.model.layers[:-3]:
                    layer.trainable = False
                logger.info("Loaded pretrained model and froze selected layers.")
            else:
                self.model = Sequential()
                self.model.add(Input(shape=input_shape))
                for layer in self.model_config['layers']:
                    layer_class = layer_mapping.get(layer['type'])
                    if layer_class:
                        if 'units' in layer:
                            # Layers like Dense, LSTM, GRU
                            self.model.add(layer_class(units=layer['units'], activation=layer['activation'], return_sequences=layer.get('return_sequences', False)))
                        else:
                            # Layers like Dropout, BatchNormalization
                            self.model.add(layer_class(rate=layer.get('rate')) if 'rate' in layer else layer_class())
                logger.info("Initialized new model.")

            optimizer = Adam(learning_rate=self.model_config.get('optimizer', {}).get('learning_rate', 0.001))
            self.model.compile(optimizer=optimizer, loss=self.model_config.get('loss', 'mse'))
            logger.info(f"Compiled model for {self.ticker} with optimizer and loss.")


    def train(self, X_train, y_train, X_val, y_val):
        # Reshape data to include time steps dimension for GRU
        X_train = np.expand_dims(X_train, axis=1)  # Adding time dimension
        X_val = np.expand_dims(X_val, axis=1)

        try:
            logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")

            self.build_model(X_train.shape[1:])

            early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_config.get('patience', 20), restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(f"best_model_{self.ticker}.keras", save_best_only=True, monitor='val_loss', mode='min')
            tensorboard = TensorBoard(log_dir=f"logs/{self.ticker}")
            lr_scheduler = LearningRateScheduler(self.scheduler)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

            callbacks = [early_stopping, model_checkpoint, tensorboard, lr_scheduler, reduce_lr]
            logger.info("Initialized callbacks.")

            self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, callbacks=callbacks)
            logger.info(f"Model training for {self.ticker} completed.")

            y_pred_val = self.model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            logger.info(f"{self.ticker} - Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            return self.model
        except Exception as e:
            logger.error(f"Error during model training for {self.ticker}.", exc_info=True)
            raise e

class NNModelConfig:
    @staticmethod
    def dense_model():
        return {
            'layers': [
                {'type': 'dense', 'units': 128, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'batch_norm'},
                {'type': 'dense', 'units': 64, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'dense', 'units': 32, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }

    @staticmethod
    def lstm_model():
        return {
            'layers': [
                {'type': 'lstm', 'units': 100, 'activation': 'tanh', 'return_sequences': True, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'lstm', 'units': 100, 'activation': 'tanh', 'return_sequences': False, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'dense', 'units': 50, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }

    @staticmethod
    def gru_model():
        return {
            'layers': [
                {'type': 'gru', 'units': 100, 'activation': 'tanh', 'return_sequences': True, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'gru', 'units': 100, 'activation': 'tanh', 'return_sequences': False, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'dense', 'units': 50, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }

def main():
    data_loader = DataLoader(logger)
    stock_tickers = ['AAPL', 'GOOG', 'MSFT', 'TSLA']
    data_directory = Path("C:/TheTradingRobotPlug/data/alpha_vantage")

    for ticker in stock_tickers:
        data_file_path = data_directory / f"{ticker.lower()}_data.csv"
        if not os.path.exists(data_file_path):
            logger.error(f"Data file not found: {data_file_path}")
            continue

        data = data_loader.load_data(data_file_path)
        if data is None:
            logger.error(f"Failed to load data for {ticker}. Skipping.")
            continue

        config_manager = ConfigManager()
        data_preprocessor = DataPreprocessor(logger, config_manager)

        # Define which columns to use based on user selection or other logic
        feature_columns = [col for col in data.columns if col not in ['date', 'symbol']]
        target_column = 'close'  # Assuming 'close' is the target variable

        X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(
            data,
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=0.2,  # Example test size
            random_state=42  # Ensures reproducibility
        )

        if X_train is None or X_val is None:
            logger.error(f"Data preprocessing failed for {ticker}. Skipping.")
            continue

        model_config = NNModelConfig.gru_model()  # Or choose another model configuration like `lstm_model` or `dense_model`
        trainer = NeuralNetworkTrainer(model_config=model_config, epochs=50, ticker=ticker)
        model = trainer.train(X_train, y_train, X_val, y_val)

        # Optionally, save the model, predictions, or further analyze the results
        model_save_path = Path(f"models/{ticker}_model.h5")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
