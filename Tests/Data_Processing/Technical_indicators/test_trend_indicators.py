# C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_trend_indicators.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Processing.Technical_indicators.trend_indicators import TrendIndicators

class TestTrendIndicators(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'high': pd.Series(np.random.rand(100) * 100),
            'low': pd.Series(np.random.rand(100) * 100),
            'close': pd.Series(np.random.rand(100) * 100),
            'volume': pd.Series(np.random.randint(1, 1000, 100))
        }
        self.df = pd.DataFrame(self.data)

    def test_add_moving_average(self):
        df_result_sma = TrendIndicators.add_moving_average(self.df.copy(), ma_type='SMA')
        self.assertIn('SMA_10', df_result_sma.columns)
        self.assertFalse(df_result_sma['SMA_10'].isna().all())

        df_result_ema = TrendIndicators.add_moving_average(self.df.copy(), ma_type='EMA')
        self.assertIn('EMA_10', df_result_ema.columns)
        self.assertFalse(df_result_ema['EMA_10'].isna().all())

    def test_calculate_macd_components(self):
        df_result = TrendIndicators.calculate_macd_components(self.df.copy())
        self.assertIn('MACD', df_result.columns)
        self.assertIn('MACD_Signal', df_result.columns)
        self.assertIn('MACD_Hist', df_result.columns)
        self.assertIn('MACD_Hist_Signal', df_result.columns)
        self.assertFalse(df_result['MACD'].isna().all())
        self.assertFalse(df_result['MACD_Signal'].isna().all())
        self.assertFalse(df_result['MACD_Hist'].isna().all())
        self.assertFalse(df_result['MACD_Hist_Signal'].isna().all())

    def test_add_adx(self):
        df_result = TrendIndicators.add_adx(self.df.copy())
        self.assertIn('ADX', df_result.columns)
        self.assertFalse(df_result['ADX'].isna().all())

    def test_add_ichimoku_cloud(self):
        df_result = TrendIndicators.add_ichimoku_cloud(self.df.copy())
        self.assertIn('Ichimoku_Conversion_Line', df_result.columns)
        self.assertIn('Ichimoku_Base_Line', df_result.columns)
        self.assertIn('Ichimoku_Leading_Span_A', df_result.columns)
        self.assertIn('Ichimoku_Leading_Span_B', df_result.columns)
        self.assertIn('Ichimoku_Lagging_Span', df_result.columns)
        self.assertFalse(df_result['Ichimoku_Conversion_Line'].isna().all())
        self.assertFalse(df_result['Ichimoku_Base_Line'].isna().all())
        self.assertFalse(df_result['Ichimoku_Leading_Span_A'].isna().all())
        self.assertFalse(df_result['Ichimoku_Leading_Span_B'].isna().all())
        self.assertFalse(df_result['Ichimoku_Lagging_Span'].isna().all())

    def test_add_parabolic_sar(self):
        df_result = TrendIndicators.add_parabolic_sar(self.df.copy())
        self.assertIn('PSAR', df_result.columns)
        self.assertFalse(df_result['PSAR'].isna().all())

if __name__ == '__main__':
    unittest.main()
