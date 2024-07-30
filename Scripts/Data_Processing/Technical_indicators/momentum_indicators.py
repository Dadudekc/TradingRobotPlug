# File: momentum_indicators.py
# Location: Scripts/Data_Processing/Technical_indicators
# Description: This script provides momentum indicators such as Stochastic Oscillator, RSI, Williams %R, ROC, and TRIX.

import os
import sys
import logging
import pandas as pd
from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator, ROCIndicator
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
log_file = os.path.join(log_path, 'momentum_indicators.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditional imports based on execution context
try:
    from Scripts.Utilities.config_handling import ConfigManager
except ImportError:
    from unittest.mock import Mock as ConfigManager

class MomentumIndicators:
    @staticmethod
    def add_stochastic_oscillator(df, window_size=14, user_defined_window=None):
        logger.info(f"Adding Stochastic Oscillator with window size {window_size}")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['low', 'high', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        if user_defined_window is not None:
            if not isinstance(user_defined_window, int) or user_defined_window <= 0:
                raise ValueError("User defined window size must be a positive integer.")
            window_size = user_defined_window
        elif not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        lowest_low = df['low'].rolling(window=window_size, min_periods=1).min()
        highest_high = df['high'].rolling(window=window_size, min_periods=1).max()

        # Handling division by zero
        denominator = highest_high - lowest_low
        denominator[denominator == 0] = 1

        df['Stochastic'] = 100 * ((df['close'] - lowest_low) / denominator)
        df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3, min_periods=1).mean()

        logger.info("Successfully added Stochastic Oscillator")
        return df

    @staticmethod
    def add_relative_strength_index(df, window=14, user_defined_window=None, calculation_type="default"):
        logger.info(f"Adding RSI with window {window} and calculation type {calculation_type}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        if calculation_type == "custom":
            avg_gain = gain.rolling(window=window_size).mean()
            avg_loss = loss.rolling(window=window_size).mean()
        else:  # Default calculation
            avg_gain = gain.rolling(window=window_size, min_periods=1).mean()
            avg_loss = loss.rolling(window=window_size, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df['RSI'] = rsi.fillna(0)

        logger.info("Successfully added RSI")
        return df

    @staticmethod
    def add_williams_r(df, window=14, user_defined_window=None):
        logger.info(f"Adding Williams %R with window size {window}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        highest_high = df['high'].rolling(window=window_size, min_periods=1).max()
        lowest_low = df['low'].rolling(window=window_size, min_periods=1).min()

        # Handling division by zero
        denominator = highest_high - lowest_low
        denominator[denominator == 0] = 1

        df['Williams_R'] = -100 * (highest_high - df['close']) / denominator
        df['Williams_R'] = df['Williams_R'].fillna(0)

        logger.info("Successfully added Williams %R")
        return df

    @staticmethod
    def add_rate_of_change(df, window=10, user_defined_window=None):
        logger.info(f"Adding Rate of Change with window size {window}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        # Calculating ROC
        shifted_close = df['close'].shift(window_size)
        df['ROC'] = ((df['close'] - shifted_close) / shifted_close) * 100
        df['ROC'] = df['ROC'].fillna(0)

        logger.info("Successfully added Rate of Change")
        return df

    @staticmethod
    def add_trix(df, span=15, signal_line_span=9):
        logger.info(f"Adding TRIX with span {span} and signal line span {signal_line_span}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        if not isinstance(span, int) or span <= 0 or not isinstance(signal_line_span, int) or signal_line_span <= 0:
            raise ValueError("Span and signal line span must be positive integers.")

        # Calculate the first EMA
        ema1 = df['close'].ewm(span=span, adjust=False).mean()

        # Calculate the second EMA
        ema2 = ema1.ewm(span=span, adjust=False).mean()

        # Calculate the third EMA
        ema3 = ema2.ewm(span=span, adjust=False).mean()

        # Calculate TRIX
        df['TRIX'] = 100 * (ema3.pct_change())

        # Calculate the signal line (EMA of TRIX)
        df['TRIX_signal'] = df['TRIX'].ewm(span=signal_line_span, adjust=False).mean()

        logger.info("Successfully added TRIX")
        return df

# Example usage of MomentumIndicators
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'high': pd.Series(range(100, 200)),
        'low': pd.Series(range(50, 150)),
        'close': pd.Series(range(75, 175))
    }
    df = pd.DataFrame(data)

    # Initialize MomentumIndicators
    indicators = MomentumIndicators()

    # Apply and print each indicator
    df = indicators.add_stochastic_oscillator(df)
    print("Stochastic Oscillator:\n", df[['date', 'Stochastic', 'Stochastic_Signal']].head(10))

    df = indicators.add_relative_strength_index(df)
    print("RSI:\n", df[['date', 'RSI']].head(10))

    df = indicators.add_williams_r(df)
    print("Williams %R:\n", df[['date', 'Williams_R']].head(10))

    df = indicators.add_rate_of_change(df)
    print("Rate of Change:\n", df[['date', 'ROC']].head(10))

    df = indicators.add_trix(df)
    print("TRIX:\n", df[['date', 'TRIX', 'TRIX_signal']].head(10))
