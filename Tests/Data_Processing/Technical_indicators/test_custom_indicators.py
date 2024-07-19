# C:\TheTradingRobotPlug\Tests\Data_Processing\Technical_indicators\test_custom_indicators.py

import unittest
import pandas as pd
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Processing.Technical_indicators.custom_indicators import CustomIndicators

class TestCustomIndicators(unittest.TestCase):

    def setUp(self):
        self.data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'close': pd.Series(range(100))
        }
        self.df = pd.DataFrame(self.data)

    def test_add_custom_indicator(self):
        df_result = CustomIndicators.add_custom_indicator(self.df.copy(), 'Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, window_size=5)
        self.assertIn('Sample_Custom_Indicator', df_result.columns)
        self.assertFalse(df_result['Sample_Custom_Indicator'].isna().all())

    def test_add_multiple_custom_indicators(self):
        indicators = [
            ('Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, [5], {}),
            ('Another_Custom_Indicator', CustomIndicators.another_custom_indicator, [10], {})
        ]
        df_result = CustomIndicators.add_multiple_custom_indicators(self.df.copy(), indicators)
        self.assertIn('Sample_Custom_Indicator', df_result.columns)
        self.assertIn('Another_Custom_Indicator', df_result.columns)
        self.assertFalse(df_result['Sample_Custom_Indicator'].isna().all())
        self.assertFalse(df_result['Another_Custom_Indicator'].isna().all())

    def test_validate_indicator(self):
        valid = CustomIndicators.validate_indicator(CustomIndicators.sample_custom_indicator)
        self.assertTrue(valid)

        invalid = CustomIndicators.validate_indicator(lambda x: x.non_existent_method())
        self.assertFalse(invalid)

    def test_sample_custom_indicator(self):
        series_result = CustomIndicators.sample_custom_indicator(self.df, window_size=5)
        self.assertIsInstance(series_result, pd.Series)
        self.assertEqual(len(series_result), len(self.df))

    def test_another_custom_indicator(self):
        series_result = CustomIndicators.another_custom_indicator(self.df, window_size=10)
        self.assertIsInstance(series_result, pd.Series)
        self.assertEqual(len(series_result), len(self.df))

    def test_memory_cache(self):
        result = CustomIndicators.memory_cache('test_key', CustomIndicators.sample_custom_indicator, self.df, window_size=5)
        self.assertIsInstance(result, pd.Series)
        self.assertIn('test_key', CustomIndicators._cache)

    def test_file_cache(self):
        cache_key = 'test_file_cache_key'
        cache_file = f"cache/{cache_key}.pkl"
        
        if os.path.exists(cache_file):
            os.remove(cache_file)

        result = CustomIndicators.file_cache(cache_key, CustomIndicators.sample_custom_indicator, self.df, window_size=5)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(os.path.exists(cache_file))

        if os.path.exists(cache_file):
            os.remove(cache_file)

if __name__ == '__main__':
    unittest.main()
