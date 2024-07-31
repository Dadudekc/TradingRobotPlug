# C:\TheTradingRobotPlug\Scripts\Utilities\DataHandler.py

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import traceback
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os
import sys
from pathlib import Path
import logging
import optuna
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization, Attention
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.config_handling import ConfigManager
from Scripts.Utilities.data_store import DataStore

class DataHandler:
    def __init__(self, log_text_widget=None, data_store=None, logger=None):
        # Initialize ConfigManager
        self.config_manager = ConfigManager()
        
        self.config = self.config_manager.config
        self.log_text_widget = log_text_widget
        self.data_store = data_store
        self.logger = logger or logging.getLogger(__name__)
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        self.log("DataHandler initialized.")

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            self.log(f"Data loaded from {file_path}.")
            return data
        except Exception as e:
            error_message = f"Failed to load data from {file_path}: {str(e)}"
            self.log(error_message, "ERROR")
            return None

    def preprocess_data(self, data, target_column='close', date_column='date', lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20], scaler_type=None):
        try:
            if isinstance(data, str):
                raise ValueError("Expected data to be a DataFrame, got string instead.")

            # Convert all date columns to numerical values
            for col in data.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
                data[col] = pd.to_numeric(data[col].view('int64'))

            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                reference_date = data[date_column].min()
                data['days_since_reference'] = (data[date_column] - reference_date).dt.days
                data.drop(columns=[date_column], inplace=True)

            if 'index' not in data.columns:
                data.reset_index(inplace=True, drop=True)

            data = self.create_lag_features(data, target_column, lag_sizes)
            data = self.create_rolling_window_features(data, target_column, window_sizes)

            if data.dropna().empty:
                self.log("The dataset became empty after creating lag and rolling window features due to NaN removal. Please adjust the lag and window sizes.", "ERROR")
                return None, None, None, None
            else:
                data.dropna(inplace=True)

            if target_column in data.columns:
                y = data[target_column]
                X = data.drop(columns=[target_column], errors='ignore')
            else:
                self.log(f"The '{target_column}' column is missing from the dataset. Please check the dataset.", "ERROR")
                return None, None, None, None

            # Convert non-numeric data to NaN
            X = X.apply(pd.to_numeric, errors='coerce')

            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            if scaler_type is None:
                scaler_type = self.config_manager.get('SCALING_DEFAULT_SCALER') or 'StandardScaler'
            scaler = self.scalers.get(scaler_type, StandardScaler())
            X_scaled = scaler.fit_transform(X_imputed)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            self.log("Data preprocessing completed.")
            self.log(f"Features after preprocessing: {list(X.columns)}")
            self.log(f"Types of features: {dict(X.dtypes)}")
            
            # Additional logging to debug the issue
            self.log(f"X_train types: {dict(pd.DataFrame(X_train).dtypes)}")
            self.log(f"y_train type: {y_train.dtype}")
            
            return X_train, X_val, y_train, y_val
        except Exception as e:
            error_message = f"Error during data preprocessing: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
            return None, None, None, None

    def create_lag_features(self, df, column_name, lag_sizes):
        if column_name not in df.columns:
            self.log(f"Warning: Column '{column_name}' not found in DataFrame. Skipping lag feature creation.", "ERROR")
            return df

        for lag_days in lag_sizes:
            df[f'{column_name}_lag_{lag_days}'] = df[column_name].shift(lag_days)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.log(f"Lag features created for column '{column_name}' with lag sizes {lag_sizes}.")
        return df

    def create_rolling_window_features(self, data, column_name, windows, method='pad'):
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()

            if method == 'interpolate':
                data[f'{column_name}_rolling_mean_{window}'].interpolate(method='linear', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].interpolate(method='linear', inplace=True)
            elif method == 'pad':
                data[f'{column_name}_rolling_mean_{window}'].fillna(method='pad', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].fillna(method='pad', inplace=True)
            else:
                data.fillna(data.mean(), inplace=True)

        self.log(f"Rolling window features created for column '{column_name}' with window sizes {windows}.")
        return data

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.log(f"Data split into training and validation sets with test size {test_size}.")
        return X_train, X_val, y_train, y_val

    def scale_data(self, X_train, X_val, scaler_type='StandardScaler'):
        scaler = self.scalers.get(scaler_type, StandardScaler())
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.log(f"Data scaled using {scaler_type}.")
        return X_train_scaled, X_val_scaled, scaler

    def save_scaler(self, scaler, file_path):
        joblib.dump(scaler, file_path)
        self.log(f"Scaler saved to {file_path}.")

    def load_scaler(self, file_path):
        try:
            scaler = joblib.load(file_path)
            self.log(f"Scaler loaded from {file_path}.")
            return scaler
        except Exception as e:
            self.log(f"Failed to load scaler from {file_path}: {str(e)}", "ERROR")
            return None

    def save_metadata(self, metadata, file_path):
        with open(file_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        self.log(f"Metadata saved to {file_path}.")

    def load_metadata(self, file_path):
        try:
            with open(file_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.log(f"Metadata loaded from {file_path}.")
            return metadata
        except Exception as e:
            self.log(f"Failed to load metadata from {file_path}: {str(e)}", "ERROR")
            return None

    def plot_confusion_matrix(self, y_true=None, y_pred=None, conf_matrix=None, class_names=None, save_path="confusion_matrix.png", show_plot=True):
        if conf_matrix is None:
            if y_true is None or y_pred is None:
                self.log("You must provide either a confusion matrix or true and predicted labels.", "ERROR")
                return
            conf_matrix = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = list(range(conf_matrix.shape[0]))

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        self.log(f"Confusion matrix plot saved to {save_path}.")

    def preview_data(self, file_path):
        if not os.path.exists(file_path):
            self.log(f"File does not exist: {file_path}", "ERROR")
            return

        if not file_path.endswith('.csv'):
            self.log("Unsupported file format. Only CSV files are supported.", "ERROR")
            return

        try:
            data = pd.read_csv(file_path)
            self.log(f"Data preview from {file_path}:\n{data.head()}")
        except Exception as e:
            self.log(f"An error occurred while reading the file: {str(e)}", "ERROR")

    def fetch_and_preprocess_data(self, symbol, target_column='close', date_column='date', lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20], scaler_type='StandardScaler'):
        try:
            if not self.data_store:
                self.log("DataStore is not initialized.", "ERROR")
                return None, None, None, None

            data = self.data_store.load_data(symbol)
            
            if data is None:
                self.log(f"No data available for {symbol}.", "ERROR")
                return None, None, None, None

            # Ensure the data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                self.log("Loaded data is not a DataFrame.", "ERROR")
                return None, None, None, None

            return self.preprocess_data(data, target_column, date_column, lag_sizes, window_sizes, scaler_type)
        except Exception as e:
            error_message = f"Error during fetch and preprocess data: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
            return None, None, None, None

class LSTMModelTrainer:
    def __init__(self, logger, model_save_path='best_model.keras', scaler_save_path='scaler.pkl'):
        self.logger = logger
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path

    def preprocess_data(self, X_train, X_val):
        """Preprocess data by handling missing values and scaling."""
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        joblib.dump(scaler, self.scaler_save_path)  # Save the scaler
        return X_train_scaled, X_val_scaled

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance on the test dataset."""
        self.logger.info("Evaluating model on test data...")
        try:
            # Load the model
            model = load_model(self.model_save_path)
            scaler = joblib.load(self.scaler_save_path)

            # Preprocess the test data
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            # Make predictions
            y_pred_test = model.predict(X_test_scaled).flatten()

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)

            self.logger.info(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")

    @staticmethod
    def create_sequences(data, target, time_steps=10):
        xs, ys = [], []
        for i in range(len(data) - time_steps):
            x = data[i:(i + time_steps)]
            y = target.iloc[i + time_steps]  # Ensure correct indexing with iloc
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train_lstm(self, X_train, y_train, X_val, y_val, model, epochs=100):
        """Train an LSTM model."""
        self.logger.info("Starting LSTM model training...")
        try:
            # Log shapes of the input data
            self.logger.info(f"X_train shape before training: {X_train.shape}")
            self.logger.info(f"X_val shape before training: {X_val.shape}")

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True)
            lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32,
                      callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler])

            y_pred_val = model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            return model

        except Exception as e:
            self.logger.error(f"Error occurred during model training: {e}")
            return None

    def objective(self, trial, X_train, y_train, X_val, y_val):
        model_config = {
            'layers': [
                {'type': 'bidirectional_lstm', 'units': trial.suggest_int('units_lstm', 50, 200), 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                {'type': 'attention'},
                {'type': 'batch_norm'},
                {'type': 'dropout', 'rate': trial.suggest_float('dropout_rate', 0.2, 0.5)},
                {'type': 'dense', 'units': trial.suggest_int('units_dense', 50, 200), 'activation': 'relu'},
                {'type': 'output', 'units': 1}
            ],
            'optimizer': 'adam',
            'loss': 'mean_squared_error'
        }

        model = self.build_model(model_config)

        # Training the model
        model = self.train_lstm(X_train, y_train, X_val, y_val, model, epochs=50)

        # Validation performance
        y_pred_val = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        return mse

    def build_model(self, config):
        model = Sequential()
        for layer in config['layers']:
            if layer['type'] == 'bidirectional_lstm':
                model.add(Bidirectional(LSTM(units=layer['units'], return_sequences=layer.get('return_sequences', False),
                                             kernel_regularizer=layer.get('kernel_regularizer', None))))
            elif layer['type'] == 'attention':
                model.add(Attention())
            elif layer['type'] == 'batch_norm':
                model.add(BatchNormalization())
            elif layer['type'] == 'dropout':
                model.add(Dropout(rate=layer['rate']))
            elif layer['type'] == 'dense':
                model.add(Dense(units=layer['units'], activation=layer.get('activation', 'relu')))
            elif layer['type'] == 'output':
                model.add(Dense(units=layer['units']))

        model.compile(optimizer=Adam(), loss=config['loss'])
        return model

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        import optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Assuming X_train, y_train, X_val, y_val are already defined and preprocessed
    X_train, y_train, X_val, y_val = np.random.rand(100, 10, 5), np.random.rand(100), np.random.rand(20, 10, 5), np.random.rand(20)

    trainer = LSTMModelTrainer(logger)
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)

    # Assuming test data is available
    X_test, y_test = np.random.rand(20, 10, 5), np.random.rand(20)
    trainer.evaluate_model(X_test, y_test)
