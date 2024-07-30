# File: volume_indicators.py
# Location: Scripts/Data_Processing/Technical_indicators
# Description: This script provides volume indicators such as Money Flow Index, On-Balance Volume, VWAP, Accumulation/Distribution Line, Chaikin Money Flow, and Volume Oscillator.

import os
import sys
import logging
import pandas as pd
import numpy as np
from ta.volume import MFIIndicator
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
log_file = os.path.join(log_path, 'volume_indicators.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditional imports based on execution context
try:
    from Scripts.Utilities.config_handling import ConfigManager
except ImportError:
    from unittest.mock import Mock as ConfigManager

class VolumeIndicators:
    @staticmethod
    def add_money_flow_index(df, window=14, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close', 'volume']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window_size, min_periods=1).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window_size, min_periods=1).sum()

        # Handling division by zero
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))

        df['MFI'] = mfi.fillna(0)
        return df

    @staticmethod
    def add_on_balance_volume(df, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not all(column in df.columns for column in ['volume', 'close']):
            raise ValueError("DataFrame must contain 'volume' and 'close' columns")

        window_size = user_defined_window if user_defined_window is not None else 14
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        obv_change = df['volume'] * np.sign(df['close'].diff()).fillna(0)
        df['OBV'] = obv_change.cumsum()

        return df

    @staticmethod
    def add_vwap(df, user_defined_window=None, price_type='typical', adjust_for_splits=False):
        start_time = timer()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close', 'volume']
        if adjust_for_splits and 'split_coefficient' not in df.columns:
            raise ValueError("DataFrame must contain a 'split_coefficient' column for split adjustments.")
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = len(df) if user_defined_window is None else user_defined_window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        if price_type == 'typical':
            prices = (df['high'] + df['low'] + df['close']) / 3
        elif price_type == 'weighted_close':
            prices = (df['high'] + df['low'] + 2 * df['close']) / 4
        else:
            raise ValueError("Invalid price type specified.")

        if adjust_for_splits:
            adjusted_volume = df['volume'] / df['split_coefficient']
        else:
            adjusted_volume = df['volume']

        vwap = (prices * adjusted_volume).cumsum() / adjusted_volume.cumsum()
        df['VWAP'] = vwap

        execution_time = timer() - start_time
        logger.info(f"VWAP calculation completed in {execution_time:.2f} seconds.")

        return df

    @staticmethod
    def add_accumulation_distribution_line(df, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['close', 'low', 'high', 'volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        high_low_diff = df['high'] - df['low']
        high_low_diff.replace(to_replace=0, method='ffill', inplace=True)  # Prevent division by zero

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
        clv.fillna(0, inplace=True)  # Handle NaN values
        adl = (clv * df['volume']).cumsum()
        df['ADL'] = adl

        logger.debug(f"High-Low Diff: {high_low_diff.head(10)}")
        logger.debug(f"CLV values: {clv.head(10)}")
        logger.debug(f"ADL values: {adl.head(10)}")

        return df

    @staticmethod
    def add_chaikin_money_flow(df, user_defined_window=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window = user_defined_window if user_defined_window is not None else 14  # Default value
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv.fillna(0, inplace=True)
        money_flow_volume = clv * df['volume']
        cmf = money_flow_volume.rolling(window=window, min_periods=1).sum() / df['volume'].rolling(window=window, min_periods=1).sum()
        df['CMF'] = cmf

        logger.debug(f"High-Low Diff: {df['high'] - df['low']}")
        logger.debug(f"CLV values: {clv.head(10)}")
        logger.debug(f"Money Flow Volume: {money_flow_volume.head(10)}")
        logger.debug(f"CMF values: {cmf.head(10)}")

        return df

    @staticmethod
    def add_volume_oscillator(df, short_window=12, long_window=26):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain a 'volume' column")

        if not isinstance(short_window, int) or not isinstance(long_window, int) or short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")
        if short_window >= long_window:
            raise ValueError("Short window size must be less than long window size.")

        short_vol_ema = df['volume'].ewm(span=short_window, adjust=False).mean()
        long_vol_ema = df['volume'].ewm(span=long_window, adjust=False).mean()
        df['Volume_Oscillator'] = short_vol_ema - long_vol_ema

        return df

# Example usage of VolumeIndicators
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'high': pd.Series(range(100, 200)),
        'low': pd.Series(range(50, 150)),
        'close': pd.Series(range(75, 175)),
        'volume': pd.Series(range(1000, 1100))
    }
    df = pd.DataFrame(data)

    # Initialize VolumeIndicators
    indicators = VolumeIndicators()

    # Apply and print each indicator
    df = indicators.add_money_flow_index(df)
    print("Money Flow Index (MFI):\n", df[['date', 'MFI']].head(10))

    df = indicators.add_on_balance_volume(df)
    print("On-Balance Volume (OBV):\n", df[['date', 'OBV']].head(10))

    df = indicators.add_vwap(df)
    print("Volume Weighted Average Price (VWAP):\n", df[['date', 'VWAP']].head(10))

    df = indicators.add_accumulation_distribution_line(df)
    print("Accumulation/Distribution Line (ADL):\n", df[['date', 'ADL']].head(10))

    df = indicators.add_chaikin_money_flow(df)
    print("Chaikin Money Flow (CMF):\n", df[['date', 'CMF']].head(10))

    df = indicators.add_volume_oscillator(df)
    print("Volume Oscillator:\n", df[['date', 'Volume_Oscillator']].head(10))
