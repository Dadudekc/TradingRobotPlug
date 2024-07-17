# C:\TheTradingRobotPlug\Scripts\Data_Processing\Technical_indicators\momentum_indicators.py
import talib
import pandas as pd
from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator, ROCIndicator
import time

class MomentumIndicators:
    @staticmethod
    def add_stochastic_oscillator(df, window_size=14, user_defined_window=None):
        """
        Adds Stochastic Oscillator and its signal line to the DataFrame.

        :param df: pandas DataFrame containing high, low, and close price data
        :param window_size: Integer, default window size for the Stochastic Oscillator
        :param user_defined_window: Integer, user-defined window size, overrides default if provided
        :return: DataFrame with Stochastic Oscillator and Signal columns added
        """
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

        return df

    @staticmethod
    def add_relative_strength_index(df, window=14, user_defined_window=None, calculation_type="default"):
        """
        Adds the Relative Strength Index (RSI) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for RSI calculation.
            user_defined_window (int): Optional, user-defined window size for RSI.
            calculation_type (str): Type of RSI calculation ('default' or 'custom').

        Returns:
            DataFrame: Modified DataFrame with the RSI column added.
        """
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
        return df

    @staticmethod
    def add_williams_r(df, window=14, user_defined_window=None):
        """
        Adds the Williams %R indicator to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for Williams %R calculation.
            user_defined_window (int): Optional, user-defined window size for Williams %R.

        Returns:
            DataFrame: Modified DataFrame with the Williams %R column added.
        """
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
        return df

    @staticmethod
    def add_rate_of_change(df, window=10, user_defined_window=None):
        """
        Adds the Rate of Change (ROC) indicator to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for ROC calculation.
            user_defined_window (int): Optional, user-defined window size for ROC.

        Returns:
            DataFrame: Modified DataFrame with the ROC column added.
        """
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

        return df

    @staticmethod
    def add_trix(df, span=15, signal_line_span=9):
        """
        Add TRIX and its signal line to the DataFrame.

        TRIX is a momentum indicator that shows the percentage change in a triple exponentially smoothed moving average.
        The signal line is an EMA of the TRIX.

        Args:
            df (DataFrame): DataFrame with a 'close' column.
            span (int): The span for calculating TRIX. Default is 15.
            signal_line_span (int): The span for calculating the signal line. Default is 9.

        Returns:
            pd.DataFrame: DataFrame with TRIX and signal line columns added.
        """
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

        return df
