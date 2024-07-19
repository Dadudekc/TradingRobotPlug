# C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_volume_indicators.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Processing.Technical_indicators.volume_indicators import VolumeIndicators

class TestVolumeIndicators(unittest.TestCase):

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

    def test_add_money_flow_index(self):
        df_result = VolumeIndicators.add_money_flow_index(self.df.copy())
        self.assertIn('MFI', df_result.columns)
        self.assertFalse(df_result['MFI'].isna().all())

    def test_add_on_balance_volume(self):
        df_result = VolumeIndicators.add_on_balance_volume(self.df.copy())
        self.assertIn('OBV', df_result.columns)
        self.assertFalse(df_result['OBV'].isna().all())

    def test_add_vwap(self):
        df_result = VolumeIndicators.add_vwap(self.df.copy())
        self.assertIn('VWAP', df_result.columns)
        self.assertFalse(df_result['VWAP'].isna().all())

    def test_add_accumulation_distribution_line(self):
        df_result = VolumeIndicators.add_accumulation_distribution_line(self.df.copy())
        self.assertIn('ADL', df_result.columns)
        self.assertFalse(df_result['ADL'].isna().all())

    def test_add_chaikin_money_flow(self):
        df_result = VolumeIndicators.add_chaikin_money_flow(self.df.copy())
        self.assertIn('CMF', df_result.columns)
        self.assertFalse(df_result['CMF'].isna().all())

    def test_add_volume_oscillator(self):
        df_result = VolumeIndicators.add_volume_oscillator(self.df.copy())
        self.assertIn('Volume_Oscillator', df_result.columns)
        self.assertFalse(df_result['Volume_Oscillator'].isna().all())

if __name__ == '__main__':
    unittest.main()
