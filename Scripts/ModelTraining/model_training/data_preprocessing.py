# C:\TheTradingRobotPlug\Scripts\ModelTraining\model_training\data_preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessing:
    def __init__(self, logger):
        self.logger = logger

    def display_message(self, message, level="INFO"):
        """Log messages with timestamps."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {message}"
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.debug(message)

    def preprocess_data_with_feature_engineering(self, data):
        """Preprocess the data by creating lag and rolling window features."""
        if data.empty:
            self.display_message("The dataset is empty before preprocessing. Please check the data source.", "ERROR")
            return None, None

        self.display_message(f"Original data shape: {data.shape}", "INFO")

        # Investigate NaN values
        nan_summary = data.isna().sum()
        self.display_message(f"NaN values per column:\n{nan_summary}", "INFO")

        # Drop non-numeric columns before processing
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if 'date' in non_numeric_cols:
            non_numeric_cols = non_numeric_cols.drop('date')
        if len(non_numeric_cols) > 0:
            data = data.drop(columns=non_numeric_cols)
            self.display_message(f"Dropped non-numeric columns: {non_numeric_cols.tolist()}", "INFO")

        data.columns = data.columns.str.replace('^[0-9]+\\. ', '', regex=True)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            reference_date = data['date'].min()
            data['days_since_reference'] = (data['date'] - reference_date).dt.days

        data = self.create_lag_features(data, 'close', [1, 2, 3, 5, 10])
        data = self.create_rolling_window_features(data, 'close', [5, 10, 20])

        self.display_message(f"Data shape after feature engineering: {data.shape}", "INFO")

        # Fill NaN values with the median of the column for numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        self.display_message(f"Data shape after filling NaNs: {data.shape}", "INFO")

        # Ensure all data is numeric
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()

        if 'close' in data.columns:
            y = data['close']
            X = data.drop(columns=['close', 'date'], errors='ignore')

            self.display_message(f"Features shape: {X.shape}", "INFO")
            self.display_message(f"Target shape: {y.shape}", "INFO")

            return X, y
        else:
            self.display_message("The 'close' column is missing from the dataset. Please check the dataset.", "ERROR")
            return None, None

    def create_lag_features(self, df, column_name, lag_sizes):
        """Create lag features for the specified column."""
        for lag in lag_sizes:
            df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
        return df

    def create_rolling_window_features(self, data, column_name, windows):
        """Create rolling window features for the specified column."""
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()
        return data
