import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from Scripts.ModelTraining.model_training.model_training_main import ModelTrainingManager

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Load the full data set that includes all possible indicators
        self.full_data = pd.read_csv('C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv')

    def prepare_data_mock(self, data):
        num_time_steps = 10
        num_features = len(data.columns) - 2  # Adjusting for 'date' and 'symbol' columns

        # Ensure the data is aligned with the number of time steps
        total_samples = (len(data) // num_time_steps) * num_time_steps
        X_train = data.drop(columns=['date', 'symbol'], errors='ignore').iloc[:total_samples].values
        X_train = X_train.reshape(-1, num_time_steps, num_features)

        y_train = data['close'].iloc[:total_samples].values.reshape(-1, num_time_steps)

        return (X_train, y_train, MagicMock())

    def test_with_all_indicators(self):
        with patch('model_training_main.pd.read_csv', return_value=self.full_data), \
             patch('model_training_main.prepare_data', return_value=self.prepare_data_mock(self.full_data)), \
             patch('model_training_main.basicLSTMModelTrainer') as MockLSTMTrainer:

            mock_trainer_instance = MockLSTMTrainer.return_value
            mock_trainer_instance.train_lstm.return_value = (MagicMock(), MagicMock())

            # Debug prints to trace the flow and data
            print("Running test_with_all_indicators...")
            print(f"Full data columns: {self.full_data.columns}")

            # Run the predictions
            generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

            # Assertions to verify the method calls
            mock_trainer_instance.train_lstm.assert_called_once()

    def test_with_subset_of_indicators(self):
        # Select a subset of columns, ensuring they exist
        subset_data = self.full_data[['close', 'volume', 'RSI', 'MACD']].copy()

        with patch('model_training_main.pd.read_csv', return_value=subset_data), \
             patch('model_training_main.prepare_data', return_value=self.prepare_data_mock(subset_data)), \
             patch('model_training_main.basicLSTMModelTrainer') as MockLSTMTrainer:

            mock_trainer_instance = MockLSTMTrainer.return_value
            mock_trainer_instance.train_lstm.return_value = (MagicMock(), MagicMock())

            # Debug prints to trace the flow and data
            print("Running test_with_subset_of_indicators...")
            print(f"Subset data columns: {subset_data.columns}")

            # Run the predictions
            generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

            # Assertions to verify the method calls
            mock_trainer_instance.train_lstm.assert_called_once()

    def test_with_no_indicators(self):
        # Create a DataFrame with only 'date' and 'symbol'
        no_indicators_data = pd.DataFrame({'date': self.full_data['date'], 'symbol': self.full_data['symbol']})

        with patch('model_training_main.pd.read_csv', return_value=no_indicators_data), \
             patch('model_training_main.prepare_data', return_value=(pd.DataFrame().values.reshape(-1, 10, 0), pd.Series(dtype=float), MagicMock())), \
             patch('model_training_main.basicLSTMModelTrainer') as MockLSTMTrainer:

            mock_trainer_instance = MockLSTMTrainer.return_value
            mock_trainer_instance.train_lstm.return_value = (MagicMock(), MagicMock())

            # Debug prints to trace the flow and data
            print("Running test_with_no_indicators...")
            print(f"No indicators data columns: {no_indicators_data.columns}")

            # Run the predictions
            generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

            # Assertions to verify the method calls
            mock_trainer_instance.train_lstm.assert_called_once()

    def test_with_missing_values(self):
        # Introduce NaNs into the 'RSI' column
        data_with_missing = self.full_data.copy()
        data_with_missing.loc[::10, 'RSI'] = pd.NA

        with patch('model_training_main.pd.read_csv', return_value=data_with_missing), \
             patch('model_training_main.prepare_data', return_value=self.prepare_data_mock(data_with_missing)), \
             patch('model_training_main.basicLSTMModelTrainer') as MockLSTMTrainer:

            mock_trainer_instance = MockLSTMTrainer.return_value
            mock_trainer_instance.train_lstm.return_value = (MagicMock(), MagicMock())

            # Debug prints to trace the flow and data
            print("Running test_with_missing_values...")
            print(f"Data with missing values in 'RSI': {data_with_missing[['RSI']].head(15)}")

            # Run the predictions
            generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

            # Assertions to verify the method calls
            mock_trainer_instance.train_lstm.assert_called_once()

    def test_with_single_indicator(self):
        # Select only the 'close' column
        single_indicator_data = self.full_data[['close']]

        with patch('model_training_main.pd.read_csv', return_value=single_indicator_data), \
             patch('model_training_main.prepare_data', return_value=self.prepare_data_mock(single_indicator_data)), \
             patch('model_training_main.basicLSTMModelTrainer') as MockLSTMTrainer:

            mock_trainer_instance = MockLSTMTrainer.return_value
            mock_trainer_instance.train_lstm.return_value = (MagicMock(), MagicMock())

            # Debug prints to trace the flow and data
            print("Running test_with_single_indicator...")
            print(f"Single indicator data columns: {single_indicator_data.columns}")

            # Run the predictions
            generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)

            # Assertions to verify the method calls
            mock_trainer_instance.train_lstm.assert_called_once()

if __name__ == '__main__':
    unittest.main()
