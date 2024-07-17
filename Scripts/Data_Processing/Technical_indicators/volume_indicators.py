import pandas as pd
import numpy as np
from ta.volume import MFIIndicator
import time


class VolumeIndicators:
    @staticmethod
    def add_money_flow_index(df, window=14, user_defined_window=None):
        """
        Adds the Money Flow Index (MFI) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data, including volume.
            window (int): Default window size for MFI calculation.
            user_defined_window (int): Optional, user-defined window size for MFI.

        Returns:
            DataFrame: Modified DataFrame with the MFI column added.
        """
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

        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window_size).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window_size).sum()

        # Handling division by zero
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))

        df['MFI'] = mfi.fillna(0)
        return df

    @staticmethod
    def add_on_balance_volume(df, user_defined_window=None):
        """
        Adds On-Balance Volume (OBV) to the DataFrame.

        OBV is a technical trading momentum indicator that uses volume flow to predict changes in stock price.

        Args:
            df (DataFrame): DataFrame containing 'volume' and 'close' price data.
            user_defined_window (int, optional): User-defined window size. If provided, overrides default window size.

        Returns:
            DataFrame: Modified DataFrame with the OBV column added.
        """
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
        """
        Adds Volume Weighted Average Price (VWAP) to the DataFrame, using vectorized operations for efficiency.

        VWAP is calculated as the sum of price multiplied by volume, divided by the total volume.

        Args:
            df (DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' data.
            user_defined_window (int, optional): Window size for VWAP calculation. Defaults to length of DataFrame if None.
            price_type (str): Type of price to use ('typical', 'weighted_close'). 'typical' uses (high+low+close)/3.
            adjust_for_splits (bool): If True, adjusts for stock splits and dividends (requires 'split_coefficient' column).

        Returns:
            DataFrame: Modified DataFrame with the VWAP column added.
        """
        start_time = time.time()

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

        vwap = (prices * adjusted_volume).rolling(window=window_size).sum() / adjusted_volume.rolling(window=window_size).sum()
        df['VWAP'] = vwap

        execution_time = time.time() - start_time
        print(f"VWAP calculation completed in {execution_time:.2f} seconds.")

        return df

    @staticmethod
    def add_accumulation_distribution_line(df, user_defined_window=None):
        """
        Adds the Accumulation/Distribution Line (ADL) to the DataFrame.

        The ADL is a volume-based indicator designed to measure the cumulative flow of money into and out of a security.

        Args:
            df (DataFrame): DataFrame with 'close', 'low', 'high', and 'volume' data.
            user_defined_window (int, optional): User-defined window size. Defaults to 20 if None.

        Returns:
            DataFrame: Modified DataFrame with the ADL column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['close', 'low', 'high', 'volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = user_defined_window if user_defined_window is not None else 20
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        high_low_diff = df['high'] - df['low']
        high_low_diff.replace(to_replace=0, method='ffill', inplace=True)  # Prevent division by zero

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
        clv.fillna(0, inplace=True)  # Handle NaN values
        df['ADL'] = (clv * df['volume']).cumsum()

        return df

    @staticmethod
    def add_chaikin_money_flow(df, user_defined_window=None):
        """
        Adds the Chaikin Money Flow (CMF) indicator to the DataFrame.

        CMF is a volume-weighted average of accumulation and distribution over a specified period.

        Args:
            df (DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' data.
            user_defined_window (int, optional): User-defined window size. If provided, overrides default window size.

        Returns:
            DataFrame: Modified DataFrame with the CMF column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window = user_defined_window if user_defined_window is not None else 14  # Default value
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv.fillna(0, inplace=True)
        money_flow_volume = clv * df['volume']
        df['CMF'] = money_flow_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()

        return df

    @staticmethod
    def add_volume_oscillator(df, short_window=12, long_window=26):
        """
        Adds the Volume Oscillator to the DataFrame.

        The Volume Oscillator measures the difference between two volume moving averages, 
        highlighting the increasing or decreasing volume trends.

        Args:
            df (DataFrame): DataFrame containing 'volume' data.
            short_window (int): Window size for the shorter volume moving average. Defaults to 12.
            long_window (int): Window size for the longer volume moving average. Defaults to 26.

        Returns:
            DataFrame: Modified DataFrame with the Volume Oscillator column added.
        """
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
