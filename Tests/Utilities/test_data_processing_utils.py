import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

# Verify if the path is correctly added
print(f"Script directory: {script_dir}")
print(f"Project root added to PYTHONPATH: {project_root}")

# Check if the module exists in the expected location
module_path = os.path.join(project_root, 'Scripts', 'Utilities', 'Data_processing_utils.py')
print(f"Checking if module exists at: {module_path}")
print(f"Module exists: {os.path.exists(module_path)}")

try:
    from Scripts.Utilities.data_processing_utils import DataValidation, DataCleaning, DataTransformation, DataHandling
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    sys.exit(1)


class TestDataProcessingUtils(unittest.TestCase):

    def setUp(self):
        self.data = {
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, 4, np.nan, 2, 1],
            'C': ['a', 'b', 'c', 'd', 'e'],
            'D': [1, 2, 2, 3, 3]
        }
        self.df = pd.DataFrame(self.data)

    def test_validate_dataframe(self):
        DataValidation.validate_dataframe(self.df, ['A', 'B', 'C'])
        with self.assertRaises(ValueError):
            DataValidation.validate_dataframe(self.df, ['A', 'B', 'X'])

    def test_check_for_nulls(self):
        self.assertTrue(DataValidation.check_for_nulls(self.df))
        df_no_nulls = self.df.dropna()
        self.assertFalse(DataValidation.check_for_nulls(df_no_nulls))

    def test_validate_column_types(self):
        DataValidation.validate_column_types(self.df, {'A': 'float64', 'C': 'object'})
        with self.assertRaises(ValueError):
            DataValidation.validate_column_types(self.df, {'A': 'int64'})

    def test_validate_column_range(self):
        DataValidation.validate_column_range(self.df.dropna(), 'A', 1, 5)
        with self.assertRaises(ValueError):
            DataValidation.validate_column_range(self.df.dropna(), 'A', 2, 5)

    def test_validate_unique_values(self):
        DataValidation.validate_unique_values(self.df, 'C')
        with self.assertRaises(ValueError):
            DataValidation.validate_unique_values(self.df, 'D')

    def test_remove_duplicates(self):
        df_with_duplicates = self.df.append(self.df.iloc[0], ignore_index=True)
        cleaned_df = DataCleaning.remove_duplicates(df_with_duplicates)
        self.assertEqual(len(cleaned_df), len(self.df))

    def test_fill_missing_values(self):
        df_filled = DataCleaning.fill_missing_values(self.df, strategy='mean')
        self.assertFalse(df_filled.isnull().values.any())
        with self.assertRaises(ValueError):
            DataCleaning.fill_missing_values(self.df, strategy='constant')

    def test_remove_outliers(self):
        df_no_outliers = DataCleaning.remove_outliers(self.df.dropna(), 'A', method='iqr')
        self.assertTrue(df_no_outliers['A'].between(df_no_outliers['A'].quantile(0.25), df_no_outliers['A'].quantile(0.75)).all())

    def test_handle_categorical_data(self):
        df_one_hot = DataCleaning.handle_categorical_data(self.df, 'C', strategy='one_hot')
        self.assertIn('C_a', df_one_hot.columns)
        df_label = DataCleaning.handle_categorical_data(self.df, 'C', strategy='label')
        self.assertIn('C', df_label.columns)
        self.assertTrue(df_label['C'].dtype == 'int32')

    def test_normalize_column(self):
        df_normalized = DataTransformation.normalize_column(self.df.copy(), 'A')
        self.assertAlmostEqual(df_normalized['A'].min(), 0)
        self.assertAlmostEqual(df_normalized['A'].max(), 1)

    def test_standardize_column(self):
        df_standardized = DataTransformation.standardize_column(self.df.copy(), 'A')
        self.assertAlmostEqual(df_standardized['A'].mean(), 0, places=5)
        self.assertAlmostEqual(df_standardized['A'].std(), 1, places=5)

    def test_log_transform_column(self):
        df_log_transformed = DataTransformation.log_transform_column(self.df.copy(), 'A')
        self.assertTrue((df_log_transformed['A'] >= 0).all())

    def test_save_to_csv(self):
        file_path = 'test.csv'
        DataHandling.save_to_csv(self.df, file_path)
        loaded_df = DataHandling.load_from_csv(file_path)
        pd.testing.assert_frame_equal(self.df, loaded_df)
        os.remove(file_path)

    def test_save_to_sql(self):
        db_uri = 'sqlite:///:memory:'
        table_name = 'test_table'
        DataHandling.save_to_sql(self.df, table_name, db_uri)
        loaded_df = DataHandling.load_from_sql(table_name, db_uri)
        pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_merge_dataframes(self):
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        merged_df = DataHandling.merge_dataframes([df1, df2], on='key')
        self.assertIn('value1', merged_df.columns)
        self.assertIn('value2', merged_df.columns)

    def test_aggregate_data(self):
        df_aggregated = DataHandling.aggregate_data(self.df, 'D', {'A': 'mean', 'B': 'sum'})
        self.assertEqual(len(df_aggregated), self.df['D'].nunique())

    def test_convert_column_types(self):
        df_converted = DataHandling.convert_column_types(self.df, {'A': 'str'})
        self.assertEqual(df_converted['A'].dtype, 'object')

    def test_batch_process_files(self):
        file_paths = ['test1.csv', 'test2.csv']
        self.df.to_csv(file_paths[0], index=False)
        self.df.to_csv(file_paths[1], index=False)
        processed_dfs = DataHandling.batch_process_files(file_paths, DataCleaning.fill_missing_values, strategy='mean')
        for df in processed_dfs:
            self.assertFalse(df.isnull().values.any())
        for file_path in file_paths:
            os.remove(file_path)

    def test_optimized_save_to_csv(self):
        file_path = 'large_test.csv'
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        DataHandling.optimized_save_to_csv(large_df, file_path)
        loaded_df = DataHandling.load_from_csv(file_path)
        pd.testing.assert_frame_equal(large_df, loaded_df)
        os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
