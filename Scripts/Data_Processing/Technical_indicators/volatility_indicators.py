import pandas as pd
from ta.volatility import BollingerBands, KeltnerChannel
import numpy as np
import talib
import time

class VolatilityIndicators:
    @staticmethod
    def add_bollinger_bands(df, window_size=10, std_multiplier=2, user_defined_window=None):
        """
        Adds Bollinger Bands to the DataFrame.

        :param df: pandas DataFrame with price data
        :param window_size: Integer, default window size for Bollinger Bands
        :param std_multiplier: Integer, standard deviation multiplier for Bollinger Bands
        :param user_defined_window: Integer, user-defined window size, overrides default if provided
        :return: DataFrame with Bollinger Bands columns added
        """
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
        """
        Adds a rolling standard deviation to the DataFrame based on the 'close' prices.

        :param df: pandas DataFrame containing stock price data.
        :param window_size: Integer, default window size for calculating standard deviation.
        :param user_defined_window: Integer, user-defined window size for standard deviation.
        :return: DataFrame: Modified DataFrame with the rolling standard deviation column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        df['Standard_Deviation'] = df['close'].rolling(window=window).std().fillna(0)
        return df

    @staticmethod
    def add_historical_volatility(df, window=20, user_defined_window=None):
        """
        Adds historical volatility to the DataFrame, calculated as the standard deviation
        of the logarithmic returns of closing prices.

        :param df: pandas DataFrame containing the 'close' prices.
        :param window: The window size for calculating volatility. Defaults to 20.
        :param user_defined_window: User-defined window size. If provided, it overrides the default.
        :return: DataFrame: DataFrame with the new 'Historical_Volatility' column.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        log_return = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['Historical_Volatility'] = log_return.rolling(window=window_size).std() * np.sqrt(window_size)

        return df

    @staticmethod
    def add_chandelier_exit(df, window=22, multiplier=3, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Chandelier Exit indicators to the DataFrame.

        :param df: pandas DataFrame containing 'high', 'low', and 'close' prices.
        :param window: The window size for calculating the indicator. Defaults to 22.
        :param multiplier: Multiplier for the ATR value. Defaults to 3.
        :param user_defined_window: User defined window size. If provided, it overrides the default.
        :param user_defined_multiplier: User defined multiplier. If provided, it overrides the default.
        :return: DataFrame: DataFrame with new 'Chandelier_Exit_Long' column.
        """
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        if user_defined_window is not None:
            window = user_defined_window
        if user_defined_multiplier is not None:
            multiplier = user_defined_multiplier

        highest_high = df['high'].rolling(window=window).max()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window)

        df['Chandelier_Exit_Long'] = highest_high - multiplier * atr

        return df

    @staticmethod
    def add_keltner_channel(df, window=20, multiplier=2, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Keltner Channel to the DataFrame.

        :param df: pandas DataFrame with 'high', 'low', and 'close' prices.
        :param window: The window size for the moving average and ATR. Defaults to 20.
        :param multiplier: Multiplier for the ATR. Defaults to 2.
        :param user_defined_window: User-defined window size. Overrides default if provided.
        :param user_defined_multiplier: User-defined multiplier. Overrides default if provided.
        :return: DataFrame: DataFrame with Keltner Channel columns added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = user_defined_window if user_defined_window is not None else window
        atr_multiplier = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        ma = df['close'].rolling(window=window_size).mean()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window_size)
        df['Keltner_Channel_High'] = ma + (atr_multiplier * atr)
        df['Keltner_Channel_Low'] = ma - (atr_multiplier * atr)
        df['Keltner_Channel_Mid'] = ma

        return df

    @staticmethod
    def add_moving_average_envelope(df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        """
        Adds Moving Average Envelope to the DataFrame.

        :param df: pandas DataFrame containing stock price data.
        :param window_size: The window size for calculating the SMA. Defaults to 10.
        :param percentage: The percentage above and below the SMA for the envelope. Defaults to 0.025 (2.5%).
        :param user_defined_window: User-defined window size for SMA. Overrides default if provided.
        :param user_defined_percentage: User-defined percentage for the envelope. Overrides default if provided.
        :return: DataFrame: DataFrame with MAE upper and lower bounds added.
        """
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

        SMA = df['close'].rolling(window=window).mean()
        df['MAE_Upper'] = SMA * (1 + envelope_percentage)
        df['MAE_Lower'] = SMA * (1 - envelope_percentage)

        return df
