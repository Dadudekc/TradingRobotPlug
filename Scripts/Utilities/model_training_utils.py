# C:\TheTradingRobotPlug\Scripts\Utilities\model_training_utils.py

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import traceback

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

def check_for_nan_inf(data):
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values.")
        
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
            # Ensure logging method is correctly selected
            if level == "INFO":
                self.logger.info(message)
            elif level == "DEBUG":
                self.logger.debug(message)
            elif level == "WARNING":
                self.logger.warning(message)
            elif level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.log(logging.INFO, message)


class DataLoader:
    def __init__(self, logger_handler):
        self.logger = logger_handler

    # File loading function in model_training_utils.py
    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            error_message = f"No such file or directory: '{file_path}'"
            self.logger.error(error_message)  # Corrected logging usage
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

    def preprocess_data(self, data, target_column='close', feature_columns=None, test_size=0.2, random_state=42):
        """
        Preprocess the data using specified feature columns, handle date columns, create lag features,
        rolling window features, and scale the data.
        """
        try:
            # Validate input data and feature_columns
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame.")

            if feature_columns is None:
                feature_columns = data.columns.tolist()  # Use all columns if none specified
            else:
                # Ensure all specified feature_columns exist in the DataFrame
                missing_cols = [col for col in feature_columns if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

            X = data[feature_columns]
            if target_column in X:
                y = X.pop(target_column)
            else:
                raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

            # Optionally handle date columns, create lags, rolling features, etc. as necessary
            # Example: Handle date if it's one of the features
            if 'date' in X.columns:
                X = self._handle_dates(X, 'date')

            # Scale features
            if 'scaler_type' in self.config_manager.config:
                scaler_type = self.config_manager.config['scaler_type']
                scaler = self.scalers.get(scaler_type, StandardScaler())
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

            # Split the data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

            return X_train, X_val, y_train, y_val
        except Exception as e:
            self.logger.error(f"Failed to preprocess data: {str(e)}", exc_info=True)
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


# Example utility functions
def setup_logger(name, log_file=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def detect_models(model_dir):
    """Detect available models in the specified directory."""
    model_types = ['arima', 'lstm', 'neural_network', 'random_forest', 'linear_regression']
    detected_models = {}
    
    for model_type in model_types:
        model_files = list(Path(model_dir).rglob(f"*{model_type}*"))
        if model_files:
            detected_models[model_type] = str(model_files[0])  # Take the first found model
    
    return detected_models


def load_model_from_file(model_type, model_path, logger):
    try:
        # Implement model loading logic here, e.g., using joblib, pickle, etc.
        model = joblib.load(model_path)  # Example; replace with actual model loading
        logger.info(f"Successfully loaded {model_type} model from {model_path}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_type} model from {model_path}: {str(e)}")
        return None


def preprocess_data(data, model_type):
    """
    Preprocesses the data to ensure it matches the expected input shape for the model type.

    Parameters:
    - data: numpy array or pandas DataFrame, the input data to preprocess.
    - model_type: str, the type of model ('lstm', 'neural_network', etc.).

    Returns:
    - data: Preprocessed data reshaped for the specific model type.
    """
    if model_type in ['lstm', 'neural_network']:
        # Determine the expected number of features for the model
        expected_num_features = 16  # Adjust this value to match your model's training setup

        # Ensure data has the correct number of features
        if data.shape[1] != expected_num_features:
            raise ValueError(f"Input data has {data.shape[1]} features, but the model expects {expected_num_features} features.")

        # Reshape data accordingly
        if model_type == 'lstm':
            # LSTM expects 3D input: [samples, time steps, features]
            data = data.reshape((data.shape[0], -1, expected_num_features))
        elif model_type == 'neural_network':
            # Neural network expects 2D input: [samples, features]
            data = data.reshape(-1, expected_num_features)

    return data



def save_predictions(predictions, model_type, output_dir, format='parquet', compress=True):
    predictions_df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = os.path.join(output_dir, model_type, timestamp)
    os.makedirs(subdir, exist_ok=True)

    if format == 'csv':
        output_path = os.path.join(subdir, f"{model_type}_predictions.csv")
        predictions_df.to_csv(output_path, index=False)
    elif format == 'json':
        output_path = os.path.join(subdir, f"{model_type}_predictions.json")
        predictions_df.to_json(output_path, orient='records')
    elif format == 'parquet':
        output_path = os.path.join(subdir, f"{model_type}_predictions.parquet")
        predictions_df.to_parquet(output_path, index=False, compression='gzip' if compress else None)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path


def save_metadata(output_dir, model_type, model_path, input_data_path, prediction_path, logger=None):
    metadata = {
        "model_type": model_type,
        "model_path": model_path,
        "input_data_path": input_data_path,
        "prediction_path": prediction_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_df = pd.DataFrame([metadata])
    summary_file = os.path.join(output_dir, "output_summary.csv")

    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        metadata_df = pd.concat([existing_df, metadata_df])

    metadata_df.to_csv(summary_file, index=False)
    if logger:
        logger.info(f"Metadata saved to {summary_file}")


def validate_predictions(predictions, logger=None):
    if any(pred is None or np.isnan(pred).any() for pred in predictions.values()):
        if logger:
            logger.warning("Some predictions contain NaN or None values.")
    else:
        if logger:
            logger.info("All predictions are valid.")


def create_sequences(data, target, time_steps=10):
    sequences = []
    targets = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        targets.append(target[i + time_steps])
    return np.array(sequences), np.array(targets)


def prepare_data(data, target_column='close', time_steps=10):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    data = data.copy()
    numeric_data = data.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    target = data[target_column].values
    sequences, targets = create_sequences(scaled_data, target, time_steps)
    
    return sequences, targets, scaler


def load_config(config_file):
    """Function to load a YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_project_root():
    """Return the project root path based on current file location."""
    return Path(__file__).resolve().parent.parent.parent

# Example of how to use these classes:
if __name__ == "__main__":
    logger_handler = LoggerHandler()
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)
    visualization_handler = VisualizationHandler(logger_handler)
    
    # Now you can use these objects in your main code to handle different tasks.
