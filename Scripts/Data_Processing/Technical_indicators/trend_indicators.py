# trend_indicators.py

import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, IchimokuIndicator, MACD
import talib
import time

class TrendIndicators:
    @staticmethod
    def add_moving_average(df, window_size=10, user_defined_window=None, column='close', ma_type='SMA'):
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
