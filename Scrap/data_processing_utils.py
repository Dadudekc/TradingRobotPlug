 C:\TheTradingRobotPlug\Scripts\Utilities\data_processing_utils.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Union
import logging
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.config_handling import ConfigManager
from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.DataHandler import DataHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataValidation:
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validates if the DataFrame contains the required columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")

    @staticmethod
    def check_for_nulls(df: pd.DataFrame) -> bool:
        """
        Checks if the DataFrame contains any null values.
        """
        return df.isnull().values.any()

    @staticmethod
    def validate_column_types(df: pd.DataFrame, column_types: Dict[str, Any]) -> None:
        """
        Validates the types of columns in the DataFrame.
        """
        for column, expected_type in column_types.items():
            if column in df.columns:
                if not pd.api.types.is_dtype_equal(df[column].dtype, expected_type):
                    raise ValueError(f"Column '{column}' is not of type {expected_type}.")
            else:
                raise ValueError(f"Column '{column}' is missing in DataFrame.")

    @staticmethod
    def validate_column_range(df: pd.DataFrame, column: str, min_val: Any, max_val: Any) -> None:
        """
        Validates if a column's values fall within a specified range.
        """
        if not df[column].between(min_val, max_val).all():
            raise ValueError(f"Column '{column}' values are not in the range {min_val} to {max_val}.")

    @staticmethod
    def validate_unique_values(df: pd.DataFrame, column: str) -> None:
        """
        Validates if a column contains unique values.
        """
        if df[column].duplicated().any():
            raise ValueError(f"Column '{column}' contains duplicate values.")

class DataCleaning:
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.
        """
        return df.drop_duplicates(subset=subset)

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', fill_value: Any = None) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame.
        """
        if strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("fill_value must be specified for 'constant' strategy.")
            return df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Removes outliers from a column in the DataFrame.
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            return df[(df[column] >= Q1 - factor * IQR) & (df[column] <= Q3 + factor * IQR)]
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            return df[(df[column] >= mean - factor * std) & (df[column] <= mean + factor * std)]
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def handle_categorical_data(df: pd.DataFrame, column: str, strategy: str = 'one_hot') -> pd.DataFrame:
        """
        Handles categorical data in a column.
        """
        if strategy == 'one_hot':
            return pd.get_dummies(df, columns=[column])
        elif strategy == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            return df
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

class DataTransformation:
    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalizes a column in the DataFrame using Min-Max scaling.
        """
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

    @staticmethod
    def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Standardizes a column in the DataFrame using Z-score normalization.
        """
        df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

    @staticmethod
    def log_transform_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies log transformation to a column in the DataFrame.
        """
        df[column] = np.log(df[column] + 1)
        return df

    @staticmethod
    def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies one-hot encoding to a column in the DataFrame.
        """
        return pd.get_dummies(df, columns=[column])

    @staticmethod
    def scale_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
        """
        Scales a column using the specified method.
        """
        if method == 'minmax':
            return DataTransformation.normalize_column(df, column)
        elif method == 'zscore':
            return DataTransformation.standardize_column(df, column)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

class DataHandling:
    @staticmethod
    def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
        """
        Saves the DataFrame to a CSV file.
        """
        df.to_csv(file_path, index=False)

    @staticmethod
    def save_to_sql(df: pd.DataFrame, table_name: str, db_uri: str) -> None:
        """
        Saves the DataFrame to a SQL table.
        """
        from sqlalchemy import create_engine
        engine = create_engine(db_uri)
        df.to_sql(table_name, engine, if_exists='replace', index=False)

    @staticmethod
    def load_from_csv(file_path: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a CSV file.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_from_sql(table_name: str, db_uri: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a SQL table.
        """
        from sqlalchemy import create_engine
        engine = create_engine(db_uri)
        return pd.read_sql(table_name, engine)

    @staticmethod
    def merge_dataframes(df_list: List[pd.DataFrame], on: str, how: str = 'inner') -> pd.DataFrame:
        """
        Merges a list of DataFrames on a specified column.
        """
        if not df_list:
            raise ValueError("The input list 'df_list' must contain at least one DataFrame.")
        merged_df = df_list[0]
        for df in df_list[1:]:
            merged_df = merged_df.merge(df, on=on, how=how)
        return merged_df

    @staticmethod
    def aggregate_data(df: pd.DataFrame, group_by: Union[str, List[str]], agg_funcs: Dict[str, Any]) -> pd.DataFrame:
        """
        Aggregates data in the DataFrame.
        """
        return df.groupby(group_by).agg(agg_funcs).reset_index()

    @staticmethod
    def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """
        Converts column types in the DataFrame.
        """
        return df.astype(column_types)

    @staticmethod
    def batch_process_files(file_paths: List[str], process_func: Any, *args, **kwargs) -> List[pd.DataFrame]:
        """
        Processes multiple files in a batch.
        """
        processed_dataframes = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            processed_df = process_func(df, *args, **kwargs)
            processed_dataframes.append(processed_df)
        return processed_dataframes

    @staticmethod
    def optimized_save_to_csv(df: pd.DataFrame, file_path: str, chunksize: int = 100000) -> None:
        """
        Saves the DataFrame to a CSV file in chunks to handle large datasets.
        """
        df.to_csv(file_path, index=False, chunksize=chunksize)

class DataProcessor:
    def __init__(self, config_file='config.ini', log_text_widget=None):
        # Initialize ConfigManager
        defaults = {
            'LOGGING': {
                'log_level': 'DEBUG'
            },
            'DIRECTORIES': {
                'data_dir': 'data/csv'
            }
        }
        self.config_manager = ConfigManager(config_file=config_file, defaults=defaults)
        
        self.data_store = DataStore()
        self.data_handler = DataHandler(config_file=config_file, log_text_widget=log_text_widget, data_store=self.data_store)
        self.log_text_widget = log_text_widget

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            print(f"[{level}] {message}")

    def validate_and_preprocess_data(self, file_path: str, required_columns: List[str], target_column: str = 'close', date_column: str = 'date', lag_sizes: List[int] = [1, 2, 3, 5, 10], window_sizes: List[int] = [5, 10, 20], scaler_type: str = 'StandardScaler'):
        """
        Loads, validates, and preprocesses the data.
        """
        try:
            df = self.data_handler.load_data(file_path)
            DataValidation.validate_dataframe(df, required_columns)
            if DataValidation.check_for_nulls(df):
                df = DataCleaning.fill_missing_values(df, strategy='mean')
            X_train, X_val, y_train, y_val = self.data_handler.preprocess_data(df, target_column, date_column, lag_sizes, window_sizes, scaler_type)
            self.log("Data validation and preprocessing completed.")
            return X_train, X_val, y_train, y_val
        except Exception as e:
            self.log(f"Error in data validation and preprocessing: {e}", "ERROR")
            return None, None, None, None

    def save_preprocessed_data(self, X_train, X_val, y_train, y_val, symbol: str):
        """
        Saves the preprocessed data.
        """
        try:
            self.data_store.save_data(pd.concat([X_train, y_train], axis=1), f"{symbol}_train")
            self.data_store.save_data(pd.concat([X_val, y_val], axis=1), f"{symbol}_val")
            self.log("Preprocessed data saved successfully.")
        except Exception as e:
            self.log(f"Error saving preprocessed data: {e}", "ERROR")

# Example of how to use the DataProcessor class
if __name__ == "__main__":
    config_file = 'config.ini'
    processor = DataProcessor(config_file=config_file)
    processor.log("DataProcessor initialized.")
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

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.config_handling import ConfigManager
from Scripts.Utilities.data_store import DataStore

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
            data = self._handle_dates(data, date_column)
            data = self._create_lag_features(data, target_column, lag_sizes)
            data = self._create_rolling_window_features(data, target_column, window_sizes)

            if target_column in data.columns:
                y = data[target_column]
                X = data.drop(columns=[target_column], errors='ignore')
            else:
                self.logger.log(f"The '{target_column}' column is missing from the dataset. Please check the dataset.", "ERROR")
                return None, None, None, None

            X = self._impute_and_scale(X, scaler_type)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

        if scaler_type is None:
            scaler_type = self.config_manager.get('SCALING_DEFAULT_SCALER') or 'StandardScaler'
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
