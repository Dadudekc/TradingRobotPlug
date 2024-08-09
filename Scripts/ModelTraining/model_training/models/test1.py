import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from advanced_lstm_trainer import AdvancedLSTMModelTrainer

class TestAdvancedLSTMModelTrainer(unittest.TestCase):

    def setUp(self):
        # Mock logger and config manager
        self.logger_handler = MagicMock()
        self.config_manager = MagicMock()
        self.trainer = AdvancedLSTMModelTrainer(self.logger_handler, self.config_manager)

    @patch('advanced_lstm_trainer.joblib.dump')  # Mocking joblib.dump
    def test_preprocess_data_with_clean_data(self, mock_joblib_dump):
        # Create a clean dataset
        data = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=100),
            'close': np.random.rand(100),
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })

        X_train, X_val, y_train, y_val = self.trainer.preprocess_data(data)

        # Assert that no NaN or infinite values exist in the processed data
        self.assertFalse(self.trainer.check_for_nan_inf(X_train))
        self.assertFalse(self.trainer.check_for_nan_inf(X_val))
        self.assertFalse(self.trainer.check_for_nan_inf(y_train))
        self.assertFalse(self.trainer.check_for_nan_inf(y_val))

    @patch('advanced_lstm_trainer.joblib.dump')  # Mocking joblib.dump
    def test_preprocess_data_with_nan_inf_values(self, mock_joblib_dump):
        # Create a dataset with NaN and infinite values
        data = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=100),
            'close': np.random.rand(100),
            'feature1': np.append([np.inf, np.nan], np.random.rand(98)),
            'feature2': np.random.rand(100)
        })

        with self.assertRaises(ValueError):
            self.trainer.preprocess_data(data)

        # Check that the logger reported the error correctly
        self.logger_handler.log.assert_any_call(logging.ERROR, "Detected NaN or infinite values in the data.")

    def test_create_sequences_with_clean_data(self):
        # Create a clean dataset
        data = np.random.rand(100, 3)
        target = np.random.rand(100)

        X_seq, y_seq = self.trainer.create_sequences(data, target, time_steps=10)

        # Check that sequences have been created correctly
        self.assertEqual(X_seq.shape[0], 90)  # 100 - 10
        self.assertEqual(X_seq.shape[1], 10)
        self.assertEqual(X_seq.shape[2], 3)
        self.assertEqual(y_seq.shape[0], 90)

        # Assert no NaN or infinite values in sequences
        self.assertFalse(self.trainer.check_for_nan_inf(X_seq))
        self.assertFalse(self.trainer.check_for_nan_inf(y_seq))

    @patch('advanced_lstm_trainer.joblib.dump')  # Mocking joblib.dump
    def test_preprocess_data_with_nan_inf_values(self, mock_joblib_dump):
        # Create a dataset with NaN and infinite values
        data = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=100),
            'close': np.random.rand(100),
            'feature1': np.append([np.inf, np.nan], np.random.rand(98)),
            'feature2': np.random.rand(100)
        })

        # Verify the presence of NaN and infinite values
        self.assertTrue(np.isinf(data['feature1']).any() or data['feature1'].isna().any())

        with self.assertRaises(ValueError):
            self.trainer.preprocess_data(data)

        # Check that the logger reported the error correctly
        self.logger_handler.log.assert_any_call(logging.ERROR, "Detected NaN or infinite values in the data.")


if __name__ == '__main__':
    unittest.main()
