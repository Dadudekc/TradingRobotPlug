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
from sklearn.metrics import mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os
import sys
from pathlib import Path

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.config_handling import ConfigManager

class DataHandler:
    def __init__(self, config_file='config.ini', log_text_widget=None, data_store=None):
        # Initialize ConfigManager
        defaults = {
            'SCALING': {
                'default_scaler': 'StandardScaler'
            },
            'DIRECTORIES': {
                'data_dir': 'data/csv'
            }
        }
        self.config_manager = ConfigManager(config_file=config_file, defaults=defaults)
        
        self.config = self.config_manager.config
        self.log_text_widget = log_text_widget
        self.data_store = data_store
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
            print(f"[{level}] {message}")

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

            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                reference_date = data[date_column].min()
                data['days_since_reference'] = (data[date_column] - reference_date).dt.days

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
                X = data.drop(columns=[target_column, date_column], errors='ignore')
            else:
                self.log(f"The '{target_column}' column is missing from the dataset. Please check the dataset.", "ERROR")
                return None, None, None, None

            # Convert non-numeric data to NaN
            X = X.apply(pd.to_numeric, errors='coerce')

            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            if scaler_type is None:
                scaler_type = self.config_manager.get('SCALING', 'default_scaler', 'StandardScaler')
            scaler = self.scalers.get(scaler_type, StandardScaler())
            X_scaled = scaler.fit_transform(X_imputed)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            self.log("Data preprocessing completed.")
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


# Example usage
if __name__ == "__main__":
    from Scripts.Utilities.data_store import DataStore

    config_file = 'config.ini'
    data_store = DataStore()
    data_handler = DataHandler(config_file=config_file, data_store=data_store)
    data_handler.preview_data('C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv')
