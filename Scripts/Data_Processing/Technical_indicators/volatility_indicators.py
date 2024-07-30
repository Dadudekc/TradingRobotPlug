# File: volatility_indicators.py
# Location: Scripts/Data_Processing/Technical_indicators
# Description: This script provides volatility indicators such as Bollinger Bands, Standard Deviation, Historical Volatility, Chandelier Exit, Keltner Channel, and Moving Average Envelope.

import os
import sys
import logging
import pandas as pd
import numpy as np
import talib
from ta.volatility import BollingerBands, KeltnerChannel
from time import time as timer

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

# Set up relative paths for resources and logs
resources_path = os.path.join(project_root, 'resources')
log_path = os.path.join(project_root, 'logs')

# Ensure the directories exist
if not os.path.exists(resources_path):
    os.makedirs(resources_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# Logging configuration
log_file = os.path.join(log_path, 'volatility_indicators.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditional imports based on execution context
try:
    from Scripts.Utilities.config_handling import ConfigManager
except ImportError:
    from unittest.mock import Mock as ConfigManager

class VolatilityIndicators:
    @staticmethod
    def add_bollinger_bands(df, window_size=10, std_multiplier=2, user_defined_window=None):
        if user_defined_window is not None:
            window_size = user_defined_window
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        rolling_mean = df['close'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window_size, min_periods=1).std().fillna(0)

        df['Bollinger_High'] = rolling_mean + (rolling_std * std_multiplier)
        df['Bollinger_Low'] = rolling_mean - (rolling_std * std_multiplier)
        df['Bollinger_Mid'] = rolling_mean

        return df

    @staticmethod
    def add_standard_deviation(df, window_size=20, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        df['Standard_Deviation'] = df['close'].rolling(window=window, min_periods=1).std().fillna(0)
        return df

    @staticmethod
    def add_historical_volatility(df, window=20, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        log_return = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['Historical_Volatility'] = log_return.rolling(window=window_size, min_periods=1).std() * np.sqrt(window_size)

        return df

    @staticmethod
    def add_chandelier_exit(df, window=22, multiplier=3, user_defined_window=None, user_defined_multiplier=None):
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        if user_defined_window is not None:
            window = user_defined_window
        if user_defined_multiplier is not None:
            multiplier = user_defined_multiplier

        highest_high = df['high'].rolling(window=window, min_periods=1).max()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window).fillna(0)

        df['Chandelier_Exit_Long'] = highest_high - multiplier * atr

        return df

    @staticmethod
    def add_keltner_channel(df, window=20, multiplier=2, user_defined_window=None, user_defined_multiplier=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = user_defined_window if user_defined_window is not None else window
        atr_multiplier = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        ma = df['close'].rolling(window=window_size, min_periods=1).mean()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window_size).fillna(0)
        df['Keltner_Channel_High'] = ma + (atr_multiplier * atr)
        df['Keltner_Channel_Low'] = ma - (atr_multiplier * atr)
        df['Keltner_Channel_Mid'] = ma

        return df

    @staticmethod
    def add_moving_average_envelope(df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        envelope_percentage = user_defined_percentage if user_defined_percentage is not None else percentage

        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if not isinstance(envelope_percentage, float) or not (0 <= envelope_percentage <= 1):
            raise ValueError("Percentage must be a float between 0 and 1.")

        SMA = df['close'].rolling(window=window, min_periods=1).mean()
        df['MAE_Upper'] = SMA * (1 + envelope_percentage)
        df['MAE_Lower'] = SMA * (1 - envelope_percentage)

        return df

# Example usage of VolatilityIndicators
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'high': pd.Series(range(100, 200)),
        'low': pd.Series(range(50, 150)),
        'close': pd.Series(range(75, 175))
    }
    df = pd.DataFrame(data)

    # Initialize VolatilityIndicators
    indicators = VolatilityIndicators()

    # Apply and print each indicator
    df = indicators.add_bollinger_bands(df)
    print("Bollinger Bands:\n", df[['date', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Mid']].head(10))

    df = indicators.add_standard_deviation(df)
    print("Standard Deviation:\n", df[['date', 'Standard_Deviation']].head(10))

    df = indicators.add_historical_volatility(df)
    print("Historical Volatility:\n", df[['date', 'Historical_Volatility']].head(10))

    df = indicators.add_chandelier_exit(df)
    print("Chandelier Exit:\n", df[['date', 'Chandelier_Exit_Long']].head(10))

    df = indicators.add_keltner_channel(df)
    print("Keltner Channel:\n", df[['date', 'Keltner_Channel_High', 'Keltner_Channel_Low', 'Keltner_Channel_Mid']].head(10))

    df = indicators.add_moving_average_envelope(df)
    print("Moving Average Envelope:\n", df[['date', 'MAE_Upper', 'MAE_Lower']].head(10))
