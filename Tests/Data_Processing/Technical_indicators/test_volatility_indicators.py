# C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_volatility_indicators.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Processing.Technical_indicators.volatility_indicators import VolatilityIndicators

class TestVolatilityIndicators(unittest.TestCase):

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

    def test_add_bollinger_bands(self):
        df_result = VolatilityIndicators.add_bollinger_bands(self.df.copy())
        self.assertIn('Bollinger_High', df_result.columns)
        self.assertIn('Bollinger_Low', df_result.columns)
        self.assertIn('Bollinger_Mid', df_result.columns)
        self.assertFalse(df_result['Bollinger_High'].isna().all())
        self.assertFalse(df_result['Bollinger_Low'].isna().all())
        self.assertFalse(df_result['Bollinger_Mid'].isna().all())

    def test_add_standard_deviation(self):
        df_result = VolatilityIndicators.add_standard_deviation(self.df.copy())
        self.assertIn('Standard_Deviation', df_result.columns)
        self.assertFalse(df_result['Standard_Deviation'].isna().all())

    def test_add_historical_volatility(self):
        df_result = VolatilityIndicators.add_historical_volatility(self.df.copy())
        self.assertIn('Historical_Volatility', df_result.columns)
        self.assertFalse(df_result['Historical_Volatility'].isna().all())

    def test_add_chandelier_exit(self):
        df_result = VolatilityIndicators.add_chandelier_exit(self.df.copy())
        self.assertIn('Chandelier_Exit_Long', df_result.columns)
        self.assertFalse(df_result['Chandelier_Exit_Long'].isna().all())

    def test_add_keltner_channel(self):
        df_result = VolatilityIndicators.add_keltner_channel(self.df.copy())
        self.assertIn('Keltner_Channel_High', df_result.columns)
        self.assertIn('Keltner_Channel_Low', df_result.columns)
        self.assertIn('Keltner_Channel_Mid', df_result.columns)
        self.assertFalse(df_result['Keltner_Channel_High'].isna().all())
        self.assertFalse(df_result['Keltner_Channel_Low'].isna().all())
        self.assertFalse(df_result['Keltner_Channel_Mid'].isna().all())

    def test_add_moving_average_envelope(self):
        df_result = VolatilityIndicators.add_moving_average_envelope(self.df.copy())
        self.assertIn('MAE_Upper', df_result.columns)
        self.assertIn('MAE_Lower', df_result.columns)
        self.assertFalse(df_result['MAE_Upper'].isna().all())
        self.assertFalse(df_result['MAE_Lower'].isna().all())

if __name__ == '__main__':
    unittest.main()
