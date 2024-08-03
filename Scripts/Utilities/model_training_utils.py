# C:\TheTradingRobotPlug\Scripts\Utilities\model_training_utils.py

import pandas as pd
import numpy as np
import joblib
import json
import traceback
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys

# Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent.parent  # Adjusted to reach the project root
utilities_dir = project_root / 'Scripts' / 'Utilities'

# Add the Utilities directory to sys.path
if utilities_dir.exists() and str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

# Print sys.path for debugging
print("Updated sys.path:")
for p in sys.path:
    print(p)

# Now attempt to import modules
try:
    from config_handling import ConfigManager
    from data_store import DataStore
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class LoggerHandler:
    def __init__(self, log_text_widget=None, logger=None):
        self.log_text_widget = log_text_widget
        self.logger = logger or logging.getLogger(__name__)

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

class DataLoader:
    def __init__(self, logger_handler):
        self.logger = logger_handler

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            self.logger.log(f"Data loaded from {file_path}.")
            return data
        except Exception as e:
            error_message = f"Failed to load data from {file_path}: {str(e)}"
            self.logger.log(error_message, "ERROR")
            return None

    def save_scaler(self, scaler, file_path):
        joblib.dump(scaler, file_path)
        self.logger.log(f"Scaler saved to {file_path}.")

    def load_scaler(self, file_path):
        try:
            scaler = joblib.load(file_path)
            self.logger.log(f"Scaler loaded from {file_path}.")
            return scaler
        except Exception as e:
            self.logger.log(f"Failed to load scaler from {file_path}: {str(e)}", "ERROR")
            return None

    def save_metadata(self, metadata, file_path):
        with open(file_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        self.logger.log(f"Metadata saved to {file_path}.")

    def load_metadata(self, file_path):
        try:
            with open(file_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.logger.log(f"Metadata loaded from {file_path}.")
            return metadata
        except Exception as e:
            self.logger.log(f"Failed to load metadata from {file_path}: {str(e)}", "ERROR")
            return None

class DataPreprocessor:
    def __init__(self, logger_handler, config_manager):
        self.logger = logger_handler
        self.config_manager = config_manager
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }

    def preprocess_data(self, data, target_column='close', date_column='date', lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20], scaler_type=None):
        try:
            if isinstance(data, str):
                raise ValueError("Expected data to be a DataFrame, got string instead.")
            
            # Handle date columns
            data = self._handle_dates(data, date_column)
            
            # Create lag features
            data = self._create_lag_features(data, target_column, lag_sizes)
            
            # Create rolling window features
            data = self._create_rolling_window_features(data, target_column, window_sizes)

            if target_column in data.columns:
                y = data[target_column]
                X = data.drop(columns=[target_column], errors='ignore')
            else:
                self.logger.log(f"The '{target_column}' column is missing from the dataset. Please check the dataset.", "ERROR")
                return None, None, None, None

            # Remove non-numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns

            # Log and drop any non-numeric columns
            if not non_numeric_columns.empty:
                self.logger.log(f"Non-numeric columns detected and will be dropped: {list(non_numeric_columns)}", "WARNING")
                X = X[numeric_columns]

            if not numeric_columns.empty:
                X_numeric = X[numeric_columns]
                X_numeric = self._impute_and_scale(X_numeric, scaler_type)
                # Convert the numpy array back to DataFrame
                X_numeric = pd.DataFrame(X_numeric, columns=numeric_columns, index=X.index)
            else:
                self.logger.log(f"No numeric columns found in the dataset for preprocessing.", "ERROR")
                return None, None, None, None

            X_processed = X_numeric

            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            self.logger.log("Data preprocessing completed.")
            return X_train, X_val, y_train, y_val
        except Exception as e:
            error_message = f"Error during data preprocessing: {str(e)}\n{traceback.format_exc()}"
            self.logger.log(error_message, "ERROR")
            return None, None, None, None

    def _handle_dates(self, data, date_column):
        if date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column])
            reference_date = data[date_column].min()
            data['days_since_reference'] = (data[date_column] - reference_date).dt.days
            data.drop(columns=[date_column], inplace=True)
        return data

    def _create_lag_features(self, df, column_name, lag_sizes):
        if column_name not in df.columns:
            self.logger.log(f"Warning: Column '{column_name}' not found in DataFrame. Skipping lag feature creation.", "ERROR")
            return df

        for lag_days in lag_sizes:
            df[f'{column_name}_lag_{lag_days}'] = df[column_name].shift(lag_days)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.logger.log(f"Lag features created for column '{column_name}' with lag sizes {lag_sizes}.")
        return df

    def _create_rolling_window_features(self, data, column_name, windows):
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()

            data.fillna(method='pad', inplace=True)

        self.logger.log(f"Rolling window features created for column '{column_name}' with window sizes {windows}.")
        return data

    def _impute_and_scale(self, X, scaler_type):
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Check if config_manager is provided and has the method get
        if self.config_manager and hasattr(self.config_manager, 'get'):
            scaler_type = scaler_type or self.config_manager.get('SCALING_DEFAULT_SCALER') or 'StandardScaler'
        else:
            scaler_type = scaler_type or 'StandardScaler'  # Default fallback

        scaler = self.scalers.get(scaler_type, StandardScaler())
        X_scaled = scaler.fit_transform(X_imputed)
        return X_scaled

class VisualizationHandler:
    def __init__(self, logger_handler):
        self.logger = logger_handler

    def plot_confusion_matrix(self, y_true=None, y_pred=None, conf_matrix=None, class_names=None, save_path="confusion_matrix.png", show_plot=True):
        if conf_matrix is None:
            if y_true is None or y_pred is None:
                self.logger.log("You must provide either a confusion matrix or true and predicted labels.", "ERROR")
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

        self.logger.log(f"Confusion matrix plot saved to {save_path}.")

# Example of how to use these classes:
if __name__ == "__main__":
    logger_handler = LoggerHandler()
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)
    visualization_handler = VisualizationHandler(logger_handler)
    
    # Now you can use these objects in your main code to handle different tasks.
