# File: trend_indicators.py
# Location: Scripts/Data_Processing/Technical_indicators
# Description: This script provides trend indicators such as Moving Averages, MACD, ADX, Ichimoku Cloud, and Parabolic SAR.

import os
import sys
import logging
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, IchimokuIndicator, MACD
import talib
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
log_file = os.path.join(log_path, 'TrendIndicators.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditional imports based on execution context
try:
    from Scripts.Utilities.config_handling import ConfigManager
except ImportError:
    from unittest.mock import Mock as ConfigManager

class TrendIndicators:
    @staticmethod
    def add_moving_average(df, window_size=10, user_defined_window=None, column='close', ma_type='SMA'):
        """
        Adds a moving average (SMA or EMA) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window_size (int): Default window size for the moving average.
            user_defined_window (int): Optional, user-defined window size for the moving average.
            column (str): Column to calculate the moving average on.
            ma_type (str): Type of moving average ('SMA' or 'EMA').

        Returns:
            DataFrame: Modified DataFrame with the moving average column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        if user_defined_window is not None:
            if not isinstance(user_defined_window, int) or user_defined_window <= 0:
                raise ValueError("User defined window size must be a positive integer.")
            window_size = user_defined_window
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if ma_type.lower() == 'sma':
            df[f'SMA_{window_size}'] = df[column].rolling(window=window_size).mean()
        elif ma_type.lower() == 'ema':
            df[f'EMA_{window_size}'] = df[column].ewm(span=window_size, adjust=False).mean()
        else:
            raise ValueError(f"Moving average type '{ma_type}' is not supported.")

        return df

    @staticmethod
    def calculate_macd_components(df, fast_period=12, slow_period=26, signal_period=9, price_column='close'):
        """
        Calculates MACD and its components (MACD line, Signal line, and Histogram) and adds them to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            fast_period (int): Fast period for MACD calculation.
            slow_period (int): Slow period for MACD calculation.
            signal_period (int): Signal period for MACD calculation.
            price_column (str): Column to calculate the MACD on.

        Returns:
            DataFrame: Modified DataFrame with MACD, Signal, Histogram, and Histogram Signal columns added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        if not all(isinstance(x, int) and x >= 0 for x in [fast_period, slow_period, signal_period]):
            raise ValueError("Period parameters must be non-negative integers.")
        if price_column not in df.columns:
            raise ValueError(f"'{price_column}' column not found in DataFrame.")

        fast_ema = df[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_column].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Signal'] = df['MACD_Hist'].ewm(span=signal_period, adjust=False).mean()

        return df

    @staticmethod
    def add_adx(df, window=14, user_defined_window=None):
        """
        Adds the Average Directional Index (ADX) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for ADX calculation.
            user_defined_window (int): Optional, user-defined window size for ADX.

        Returns:
            DataFrame: Modified DataFrame with the ADX column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        if len(df) >= window_size:
            adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window_size, fillna=True)
            df['ADX'] = adx_indicator.adx()
        else:
            df['ADX'] = pd.NA  # Filling with pandas NA for better handling

        return df

    @staticmethod
    def add_ichimoku_cloud(df, user_defined_values=None):
        """
        Adds Ichimoku Cloud components to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            user_defined_values (tuple): Optional, user-defined tuple of three integers for Ichimoku Cloud calculation.

        Returns:
            DataFrame: Modified DataFrame with Ichimoku Cloud columns added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        if user_defined_values is not None:
            if not (isinstance(user_defined_values, tuple) and len(user_defined_values) == 3):
                raise ValueError("User defined values must be a tuple of three integers.")
            nine_window, twenty_six_window, fifty_two_window = user_defined_values
        else:
            nine_window, twenty_six_window, fifty_two_window = 9, 26, 52

        def calculate_line(window):
            period_high = df['high'].rolling(window=window).max()
            period_low = df['low'].rolling(window=window).min()
            return (period_high + period_low) / 2

        df['Ichimoku_Conversion_Line'] = calculate_line(nine_window)
        df['Ichimoku_Base_Line'] = calculate_line(twenty_six_window)
        df['Ichimoku_Leading_Span_A'] = ((df['Ichimoku_Conversion_Line'] + df['Ichimoku_Base_Line']) / 2).shift(twenty_six_window)
        df['Ichimoku_Leading_Span_B'] = calculate_line(fifty_two_window).shift(twenty_six_window)
        df['Ichimoku_Lagging_Span'] = df['close'].shift(-twenty_six_window)

        return df

    @staticmethod
    def add_parabolic_sar(df, step=0.02, max_step=0.2):
        """
        Adds the Parabolic SAR indicator to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            step (float): Step increment for SAR calculation.
            max_step (float): Maximum step value for SAR calculation.

        Returns:
            DataFrame: Modified DataFrame with the PSAR column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not all(column in df.columns for column in ['high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

        psar = df['close'][0]
        psar_high = df['high'][0]
        psar_low = df['low'][0]
        bullish = True
        af = step

        psar_values = pd.Series(index=df.index)
        psar_values.iloc[0] = psar

        for i in range(1, len(df)):
            prior_psar = psar
            prior_psar_high = psar_high
            prior_psar_low = psar_low

            if bullish:
                psar = prior_psar + af * (prior_psar_high - prior_psar)
                psar_high = max(prior_psar_high, df['high'].iloc[i])
                if df['low'].iloc[i] < psar:
                    bullish = False
                    psar = prior_psar_high
                    af = step
            else:
                psar = prior_psar + af * (prior_psar_low - prior_psar)
                psar_low = min(prior_psar_low, df['low'].iloc[i])
                if df['high'].iloc[i] > psar:
                    bullish = True
                    psar = prior_psar_low
                    af = step

            if bullish:
                psar = min(psar, df['low'].iloc[i - 1])
            else:
                psar = max(psar, df['high'].iloc[i - 1])

            if (bullish and df['high'].iloc[i] > psar_high) or (not bullish and df['low'].iloc[i] < psar_low):
                af = min(af + step, max_step)

            psar_values.iloc[i] = psar

        df['PSAR'] = psar_values
        return df

# Example usage of TrendIndicators
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'high': pd.Series(range(100, 200)),
        'low': pd.Series(range(50, 150)),
        'close': pd.Series(range(75, 175))
    }
    df = pd.DataFrame(data)

    # Initialize TrendIndicators
    indicators = TrendIndicators()

    # Apply and print each indicator
    df = indicators.add_moving_average(df)
    print("Moving Average (SMA):\n", df[['date', 'SMA_10']].head(10))

    df = indicators.calculate_macd_components(df)
    print("MACD:\n", df[['date', 'MACD', 'MACD_Signal', 'MACD_Hist']].head(10))

    df = indicators.add_adx(df)
    print("ADX:\n", df[['date', 'ADX']].head(10))

    df = indicators.add_ichimoku_cloud(df)
    print("Ichimoku Cloud:\n", df[['date', 'Ichimoku_Conversion_Line', 'Ichimoku_Base_Line', 'Ichimoku_Leading_Span_A', 'Ichimoku_Leading_Span_B', 'Ichimoku_Lagging_Span']].head(10))

    df = indicators.add_parabolic_sar(df)
    print("Parabolic SAR:\n", df[['date', 'PSAR']].head(10))
