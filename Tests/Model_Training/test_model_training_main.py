# C:\TheTradingRobotPlug\Tests\ModelTraining\test_model_training_main.py
# Tests\ModelTraining\test_model_training_main.py
# To Run:
# 1st: cd C:\TheTradingRobotPlug\Tests
# 2nd: python -m unittest ModelTraining.test_model_training_main

import os
import sys
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch

# Adjust the Python path dynamically for independent execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
models_path = os.path.join(project_root, 'Scripts', 'ModelTraining', 'models')
sys.path.append(project_root)
sys.path.append(models_path)

from Scripts.ModelTraining.model_training import model_training_main

class TestModelTrainingMain(unittest.TestCase):

    @patch('Scripts.ModelTraining.model_training.model_training_main.DataStore')
    def test_load_data(self, MockDataStore):
        # Mocking DataStore
        mock_data_store = MockDataStore.return_value
        mock_data_store.fetch_from_csv.return_value = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        
        file_name = "dummy.csv"
        data = model_training_main.load_data(mock_data_store, file_name)
        
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertEqual(data['close'].tolist(), [1, 2, 3, 4, 5])
        mock_data_store.fetch_from_csv.assert_called_once_with(file_name)

    @patch('Scripts.ModelTraining.model_training.model_training_main.ARIMAModelTrainer')
    def test_train_arima_model(self, MockARIMAModelTrainer):
        mock_logger = MagicMock()
        close_prices = [1, 2, 3, 4, 5]
        
        model_training_main.train_arima_model(close_prices, mock_logger)
        
        MockARIMAModelTrainer.assert_called_once_with(close_prices, mock_logger)
        MockARIMAModelTrainer.return_value.train.assert_called_once()

    @patch('Scripts.ModelTraining.model_training.model_training_main.LinearRegressionModel')
    def test_train_linear_regression(self, MockLinearRegressionModel):
        mock_logger = MagicMock()
        X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
        
        model_training_main.train_linear_regression(X_train, y_train, X_val, y_val, mock_logger)
        
        MockLinearRegressionModel.assert_called_once_with(mock_logger)
        MockLinearRegressionModel.return_value.train.assert_called_once_with(X_train, y_train, X_val, y_val)

    @patch('Scripts.ModelTraining.model_training.model_training_main.LSTMModelTrainer')
    def test_train_lstm_model(self, MockLSTMModelTrainer):
        mock_logger = MagicMock()
        X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
        mock_model_config = MagicMock()
        
        with patch('Scripts.ModelTraining.model_training.model_training_main.ModelConfig.lstm_model', return_value=mock_model_config):
            model_training_main.train_lstm_model(X_train, y_train, X_val, y_val, mock_logger)
        
        MockLSTMModelTrainer.assert_called_once_with(mock_logger)
        MockLSTMModelTrainer.return_value.preprocess_data.assert_called_once_with(X_train, X_val)
        MockLSTMModelTrainer.return_value.train_lstm.assert_called_once_with(X_train, y_train, X_val, y_val, mock_model_config)

    @patch('Scripts.ModelTraining.model_training.model_training_main.NeuralNetworkTrainer')
    def test_train_neural_network(self, MockNeuralNetworkTrainer):
        X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
        mock_model_config = MagicMock()
        
        with patch('Scripts.ModelTraining.model_training.model_training_main.ModelConfig.dense_model', return_value=mock_model_config):
            model_training_main.train_neural_network(X_train, y_train, X_val, y_val, MagicMock())
        
        MockNeuralNetworkTrainer.assert_called_once_with(mock_model_config, epochs=50)
        MockNeuralNetworkTrainer.return_value.train.assert_called_once_with(X_train, y_train, X_val, y_val)

    @patch('Scripts.ModelTraining.model_training.model_training_main.RandomForestModel')
    def test_train_random_forest(self, MockRandomForestModel):
        mock_logger = MagicMock()
        X, y = pd.DataFrame(), pd.Series()
        
        model_training_main.train_random_forest(X, y, mock_logger)
        
        MockRandomForestModel.assert_called_once_with(logger=mock_logger)
        MockRandomForestModel.return_value.train.assert_called_once_with(X, y, random_state=42)

if __name__ == '__main__':
    unittest.main()
