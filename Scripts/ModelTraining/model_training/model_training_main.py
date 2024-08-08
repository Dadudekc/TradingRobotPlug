# File: model_training_main.py
# Location: Scripts/ModelTraining/
# Description: Main script for managing and executing various model training tasks.

import os
import sys
import logging
import pandas as pd
from time import time
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# Set up project root and add 'Utilities' to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjust based on your project structure

# Add the correct directories to sys.path
model_training_dir = project_root / 'Scripts' / 'ModelTraining' / 'model_training'
utilities_dir = project_root / 'Scripts' / 'Utilities'

sys.path.append(str(model_training_dir))
sys.path.append(str(utilities_dir))

# Importing necessary modules from Utilities and Model Trainers
from model_training_utils import (
    setup_logger, load_model_from_file, save_predictions, save_metadata, 
    validate_predictions, preprocess_data, DataPreprocessor, 
    LoggerHandler, DataLoader, detect_models
)

from data_store import DataStore
from config_handling import ConfigManager

# Importing Model Trainers
try:
    from models.basic_lstm_trainer import (
        basicLSTMModelConfig, basicLSTMModelTrainer, prepare_data
    )
    from models.advanced_lstm_trainer import AdvancedLSTMModelTrainer
    from models.arima_model_trainer import ARIMAModelTrainer
    from models.linear_regression_trainer import LinearRegressionModel
    from models.neural_network_trainer import NeuralNetworkTrainer, NNModelConfig
    from models.random_forest_trainer import RandomForestModel

except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ModelTrainingManager:
    def __init__(self, logger, config_manager):
        self.logger = logger
        self.config_manager = config_manager
        self.data_loader = DataLoader(logger)
        self.data_store = DataStore()

    def _prepare_data(self, data):
        data = data.select_dtypes(include=[float, int]).fillna(0)
        X = data.drop(columns=['close'])
        y = data['close']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        split_index = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        return X_train, X_val, y_train, y_val

    def train_arima(self, symbol, threshold=100):
        data = self.data_store.load_data(symbol)
        if data is None:
            self.logger.error(f"No data available for symbol {symbol}")
            return
        data = data.select_dtypes(exclude=['datetime', 'object'])
        arima_trainer = ARIMAModelTrainer(symbol=symbol, threshold=threshold)
        arima_trainer.train(data)

    def train_advanced_lstm(self, symbol, model_save_path="best_model.keras", scaler_save_path="scaler.pkl"):
        data = self.data_store.load_data(symbol)
        if data is None:
            self.logger.error(f"No data available for symbol {symbol}")
            return

        try:
            X_train, X_val, y_train, y_val, scaler = self._prepare_lstm_data(data)
        except ValueError as e:
            self.logger.error(f"Data preparation error: {e}")
            return

        if X_train is not None and y_train is not None:
            trainer = AdvancedLSTMModelTrainer(self.logger, self.config_manager, model_save_path, scaler_save_path)
            time_steps = 10
            X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
            X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

            self.logger.info(f"X_train_seq shape: {X_train_seq.shape}")
            self.logger.info(f"y_train_seq shape: {y_train_seq.shape}")
            self.logger.info(f"X_val_seq shape: {X_val_seq.shape}")
            self.logger.info(f"y_val_seq shape: {y_val_seq.shape}")

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            trained_model = trainer.train_lstm(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50,
                callbacks=[early_stopping, reduce_lr]
            )

            if trained_model:
                trainer.evaluate_model(X_val_seq, y_val_seq)

    def _prepare_lstm_data(self, data):
        data = data.select_dtypes(include=[float, int]).fillna(0)
        expected_features = 420
        if data.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, but got {data.shape[1]}. Check your data preparation.")
        
        X, y, scaler = prepare_data(data, target_column='close', time_steps=10)
        X = X.reshape((X.shape[0], X.shape[1], -1))
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        return X_train, X_val, y_train, y_val, scaler

    def generate_predictions(self, model_dir, data_dir, output_format='parquet', output_dir='output', parallel=False):
        detected_models = detect_models(model_dir)
        if not detected_models:
            self.logger.error("No models detected in the specified directory.")
            return

        data_file = self.data_store.list_csv_files(directory=Path(data_dir))
        if not data_file:
            self.logger.error(f"No valid data file found in directory: {data_dir}")
            return
        
        data = self.data_store.fetch_from_csv(data_file[0])
        if data is None:
            self.logger.error(f"Failed to load data from {data_file[0]}")
            return

        features = self._prepare_features(data)
        predictions = {}

        for model_type, model_path in detected_models.items():
            self.logger.info(f"Processing model type: {model_type}")
            model = load_model_from_file(model_type, model_path, self.logger)

            if model:
                preds = model.predict(preprocess_data(features.values, model_type))
                predictions[model_type] = preds
                self.logger.info(f"Predictions for {model_type}: {preds[:5]}")
            else:
                self.logger.error(f"Skipping predictions for {model_type} due to loading error.")
        
        self._save_predictions(predictions, output_dir, output_format, detected_models, data_file[0])

    def train_linear_regression(self, symbol):
        data = self.data_store.load_data(symbol)
        if data is None:
            self.logger.error(f"No data available for symbol {symbol}")
            return

        X_train, X_val, y_train, y_val = self._prepare_data(data)
        if X_train is not None and y_train is not None:
            model = LinearRegressionModel(self.logger)
            model.train_with_explainability(X_train, y_train, X_val, y_val)

    def train_neural_network(self, symbol, model_config_name="dense_model"):
        data = self.data_store.load_data(symbol)
        if data is None:
            self.logger.error(f"No data available for symbol {symbol}")
            return

        X_train, X_val, y_train, y_val = self._prepare_data(data)
        if X_train is not None and y_train is not None:
            if model_config_name == "dense_model":
                model_config = NNModelConfig.dense_model()
            elif model_config_name == "lstm_model":
                model_config = NNModelConfig.lstm_model()
            else:
                self.logger.error(f"Unknown model configuration: {model_config_name}")
                return

            trainer = NeuralNetworkTrainer(model_config=model_config, epochs=50)
            trainer.train(X_train, y_train, X_val, y_val)

    def train_random_forest(self, symbol):
        data = self.data_store.load_data(symbol)
        if data is None:
            self.logger.error(f"No data available for symbol {symbol}")
            return

        X_train, X_val, y_train, y_val = self._prepare_data(data)
        if X_train is not None and y_train is not None:
            model = RandomForestModel(self.logger)
            model.train(X_train, y_train)

    def _prepare_features(self, data):
        excluded_columns = ['date', 'symbol']
        features = data.drop(columns=[col for col in excluded_columns if col in data.columns]).select_dtypes(include=[float, int])
        return features
    
    def _save_predictions(self, predictions, output_dir, output_format, detected_models, data_file):
        for model_type, preds in predictions.items():
            prediction_path = save_predictions(preds, model_type, output_dir, format=output_format)
            save_metadata(output_dir, model_type, detected_models[model_type], data_file, prediction_path, self.logger)


if __name__ == "__main__":
    logger = setup_logger("ModelTraining")
    config_manager = ConfigManager()

    manager = ModelTrainingManager(logger, config_manager)

    start_time = time()

    # Example usage: Train Neural Network Model
    manager.train_neural_network(symbol='TSLA', model_config_name="dense_model")

    # Example usage: Train Random Forest Model
    manager.train_random_forest(symbol='TSLA')
    
    # Example usage: Train ARIMA model
    manager.train_arima(symbol="AAPL", threshold=10)

    # Example usage: Generate predictions
    manager.generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

    # Example usage: Train Advanced LSTM Model
    manager.train_advanced_lstm(symbol='TSLA')

    # Example usage: Train Linear Regression Model
    manager.train_linear_regression(symbol='TSLA')

    end_time = time()
    logger.info(f"Execution time: {end_time - start_time} seconds")
