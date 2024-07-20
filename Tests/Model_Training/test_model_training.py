import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
from model_training import ModelTraining

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.logger = MagicMock()
        self.model_training = ModelTraining(self.logger)

    def test_initialize_model_configs(self):
        configs = self.model_training.initialize_model_configs()
        self.assertIn('neural_network', configs)
        self.assertIn('LSTM', configs)
        self.assertIsInstance(configs['neural_network'], dict)
        self.assertIsInstance(configs['LSTM'], dict)

    @patch('model_training.pd.read_csv')
    def test_preprocess_data_with_feature_engineering(self, mock_read_csv):
        data = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=10),
            'close': np.random.randn(10)
        })
        mock_read_csv.return_value = data

        X_train, X_test, y_train, y_test = self.model_training.preprocess_data_with_feature_engineering(data)

        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertTrue('close_lag_1' in X_train.columns)
        self.assertTrue('close_rolling_mean_5' in X_train.columns)

    def test_display_message(self):
        self.model_training.display_message("Test message", "INFO")
        self.logger.info.assert_called_with("[{}] Test message".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def test_create_lag_features(self):
        df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        result = self.model_training.create_lag_features(df, 'close', [1, 2])
        self.assertIn('close_lag_1', result.columns)
        self.assertIn('close_lag_2', result.columns)

    def test_create_rolling_window_features(self):
        df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        result = self.model_training.create_rolling_window_features(df, 'close', [2])
        self.assertIn('close_rolling_mean_2', result.columns)
        self.assertIn('close_rolling_std_2', result.columns)

    @patch('model_training.RandomizedSearchCV')
    def test_train_linear_regression(self, mock_random_search_cv):
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = np.random.randn(100)
        X_val = pd.DataFrame(np.random.randn(20, 10))
        y_val = np.random.randn(20)

        mock_search = MagicMock()
        mock_search.best_estimator_.predict.return_value = y_val
        mock_random_search_cv.return_value = mock_search

        model = self.model_training.train_linear_regression(X_train, y_train, X_val, y_val)

        self.assertIsNotNone(model)
        self.assertTrue(mock_random_search_cv.called)
        self.logger.info.assert_called()

    @patch('model_training.RandomForestRegressor')
    @patch('model_training.RandomizedSearchCV')
    def test_train_random_forest(self, mock_random_search_cv, mock_random_forest_regressor):
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = np.random.randn(100)
        X_val = pd.DataFrame(np.random.randn(20, 10))
        y_val = np.random.randn(20)

        mock_search = MagicMock()
        mock_search.best_estimator_.predict.return_value = y_val
        mock_random_search_cv.return_value = mock_search

        model = self.model_training.train_random_forest(X_train, y_train, X_val, y_val)

        self.assertIsNotNone(model)
        self.assertTrue(mock_random_search_cv.called)
        self.logger.info.assert_called()

    @patch('model_training.Sequential')
    def test_train_neural_network_or_lstm(self, mock_sequential):
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = np.random.randn(100)
        X_val = pd.DataFrame(np.random.randn(20, 10))
        y_val = np.random.randn(20)

        mock_model = MagicMock()
        mock_sequential.return_value = mock_model

        model = self.model_training.train_neural_network_or_lstm(X_train, y_train, X_val, y_val, 'neural_network')

        self.assertIsNotNone(model)
        self.assertTrue(mock_sequential.called)
        self.logger.info.assert_called()

if __name__ == '__main__':
    unittest.main()
