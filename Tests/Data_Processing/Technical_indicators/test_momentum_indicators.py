# C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_momentum_indicators.py

import unittest
import pandas as pd
import os
import sys
import numpy as np

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Processing.Technical_indicators.momentum_indicators import MomentumIndicators

class TestMomentumIndicators(unittest.TestCase):

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

    def test_add_stochastic_oscillator(self):
        df_result = MomentumIndicators.add_stochastic_oscillator(self.df.copy())
        self.assertIn('Stochastic', df_result.columns)
        self.assertIn('Stochastic_Signal', df_result.columns)
        self.assertFalse(df_result['Stochastic'].isna().all())
        self.assertFalse(df_result['Stochastic_Signal'].isna().all())

    def test_add_relative_strength_index(self):
        df_result = MomentumIndicators.add_relative_strength_index(self.df.copy())
        self.assertIn('RSI', df_result.columns)
        self.assertFalse(df_result['RSI'].isna().all())

    def test_add_williams_r(self):
        df_result = MomentumIndicators.add_williams_r(self.df.copy())
        self.assertIn('Williams_R', df_result.columns)
        self.assertFalse(df_result['Williams_R'].isna().all())

    def test_add_rate_of_change(self):
        df_result = MomentumIndicators.add_rate_of_change(self.df.copy())
        self.assertIn('ROC', df_result.columns)
        self.assertFalse(df_result['ROC'].isna().all())

    def test_add_trix(self):
        df_result = MomentumIndicators.add_trix(self.df.copy())
        self.assertIn('TRIX', df_result.columns)
        self.assertIn('TRIX_signal', df_result.columns)
        self.assertFalse(df_result['TRIX'].isna().all())
        self.assertFalse(df_result['TRIX_signal'].isna().all())

if __name__ == '__main__':
    unittest.main()
