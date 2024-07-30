import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import logging
from io import StringIO
import sys
import os
from pathlib import Path

# Adjust the Python path dynamically for independent execution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
scripts_path = project_root / 'Scripts'
utilities_path = scripts_path / 'Utilities'
model_training_path = scripts_path / 'ModelTraining' / 'model_training'

sys.path.append(str(project_root))
sys.path.append(str(utilities_path))
sys.path.append(str(model_training_path))

# Import the functions to be tested
from model_training_main import (
    handle_missing_values, train_linear_regression, train_lstm_model,
    train_neural_network, train_random_forest, train_arima_model, main
)

class TestModelTrainingScript(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.logger = logging.getLogger('TestLogger')
        self.logger.setLevel(logging.DEBUG)
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=100),
            'symbol': ['AAPL'] * 100,
            'close': np.random.rand(100) * 100,
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'volume': np.random.randint(1000, 10000, size=100)
        })

    def tearDown(self):
        """Tear down the test environment."""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    @patch('model_training_main.ConfigManager')
    @patch('model_training_main.DataStore')
    @patch('model_training_main.DataHandler')
    def test_main(self, MockDataHandler, MockDataStore, MockConfigManager):
        """Test the main function."""
        mock_config_manager = MockConfigManager.return_value
        mock_config_manager.get.side_effect = ['some_path', 'some_key', 'some_url', 30, 'some_db', 'some_user']

        mock_data_store = MockDataStore.return_value
        mock_data_store.list_csv_files.return_value = ['sample_file.csv']

        mock_data_handler = MockDataHandler.return_value
        mock_data_handler.load_data.return_value = self.sample_data

        with patch('builtins.input', side_effect=['sample_file.csv', '1,2,3,4,5']):
            with patch('model_training_main.ThreadPoolExecutor') as MockExecutor:
                mock_executor = MockExecutor.return_value
                mock_executor.__enter__.return_value = mock_executor
                mock_executor.submit.return_value = MagicMock()
                
                main()
                
                # Check if the data was loaded correctly
                mock_data_handler.load_data.assert_called_once_with('C:/TheTradingRobotPlug/data/alpha_vantage/sample_file.csv')
                
                # Check if the appropriate models were trained
                self.assertTrue(mock_executor.submit.called)
                calls = mock_executor.submit.call_args_list
                self.assertTrue(any(call[0][0].__name__ == 'train_linear_regression' for call in calls))
                self.assertTrue(any(call[0][0].__name__ == 'train_lstm_model' for call in calls))
                self.assertTrue(any(call[0][0].__name__ == 'train_neural_network' for call in calls))
                self.assertTrue(any(call[0][0].__name__ == 'train_random_forest' for call in calls))
                self.assertTrue(any(call[0][0].__name__ == 'train_arima_model' for call in calls))

    def test_handle_missing_values(self):
        """Test handling missing values."""
        data_with_nans = self.sample_data.copy()
        data_with_nans.loc[0:10, 'close'] = np.nan
        
        result = handle_missing_values(data_with_nans)
        
        self.assertFalse(result['close'].isna().any(), "Missing values were not handled correctly.")
        
    @patch('model_training_main.LinearRegressionModel')
    def test_train_linear_regression(self, MockLinearRegressionModel):
        """Test Linear Regression training."""
        mock_model = MockLinearRegressionModel.return_value
        X_train, y_train, X_val, y_val = self._prepare_data()
        
        train_linear_regression(X_train, y_train, X_val, y_val)
        
        mock_model.train.assert_called_once_with(X_train, y_train, X_val, y_val)

    @patch('model_training_main.LSTMModelTrainer')
    def test_train_lstm_model(self, MockLSTMModelTrainer):
        """Test LSTM model training."""
        mock_trainer = MockLSTMModelTrainer.return_value
        mock_trainer.create_sequences.return_value = (np.random.rand(90, 10, 1), np.random.rand(90))
        X_train, y_train, X_val, y_val = self._prepare_data()
        
        train_lstm_model(X_train, y_train, X_val, y_val)
        
        mock_trainer.train_lstm.assert_called_once()

    @patch('model_training_main.NeuralNetworkTrainer')
    def test_train_neural_network(self, MockNeuralNetworkTrainer):
        """Test Neural Network training."""
        mock_trainer = MockNeuralNetworkTrainer.return_value
        X_train, y_train, X_val, y_val = self._prepare_data()
        
        train_neural_network(X_train, y_train, X_val, y_val)
        
        mock_trainer.train.assert_called_once_with(X_train, y_train, X_val, y_val, epochs=50)

    @patch('model_training_main.RandomForestModel')
    def test_train_random_forest(self, MockRandomForestModel):
        """Test Random Forest training."""
        mock_model = MockRandomForestModel.return_value
        X_train, y_train, _, _ = self._prepare_data()
        
        train_random_forest(X_train, y_train)
        
        mock_model.train.assert_called_once_with(X_train, y_train, random_state=42)

    @patch('model_training_main.ARIMAModelTrainer')
    def test_train_arima_model(self, MockARIMAModelTrainer):
        """Test ARIMA model training."""
        mock_trainer = MockARIMAModelTrainer.return_value
        
        train_arima_model(self.sample_data['close'])
        
        mock_trainer.train.assert_called_once()

    def _prepare_data(self):
        """Prepare sample data for model training."""
        X = self.sample_data.drop(columns=['close', 'date', 'symbol']).values.astype(np.float32)
        y = self.sample_data['close'].values.astype(np.float32)
        
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        # Reshape data for LSTM (samples, time_steps, features)
        time_steps = 10
        X_train = X_train[:(X_train.shape[0] // time_steps) * time_steps].reshape((-1, time_steps, X_train.shape[1]))
        X_val = X_val[:(X_val.shape[0] // time_steps) * time_steps].reshape((-1, time_steps, X_val.shape[1]))
        y_train = y_train[:(y_train.shape[0] // time_steps) * time_steps].reshape((-1, time_steps))
        y_val = y_val[:(y_val.shape[0] // time_steps) * time_steps].reshape((-1, time_steps))

        return X_train, y_train, X_val, y_val

if __name__ == '__main__':
    unittest.main()
