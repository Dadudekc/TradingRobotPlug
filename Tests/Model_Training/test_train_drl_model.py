import unittest
from unittest.mock import patch, MagicMock
from train_drl_model import train_drl_model

class TestTrainDRLModel(unittest.TestCase):
    @patch('train_drl_model.load_and_preprocess_data')
    @patch('train_drl_model.TradingEnv')
    @patch('train_drl_model.DummyVecEnv')
    @patch('train_drl_model.PPO')
    def test_train_drl_model(self, mock_ppo, mock_dummy_vec_env, mock_trading_env, mock_load_and_preprocess_data):
        # Mock data and scaler
        mock_data = MagicMock()
        mock_data_scaled = MagicMock()
        mock_scaler = MagicMock()
        mock_load_and_preprocess_data.return_value = (mock_data, mock_data_scaled, mock_scaler)
        
        # Mock environment
        mock_env_instance = MagicMock()
        mock_dummy_vec_env.return_value = mock_env_instance
        
        # Mock PPO model
        mock_model_instance = MagicMock()
        mock_ppo.return_value = mock_model_instance
        
        # Define file path and total timesteps
        file_path = 'dummy_path.csv'
        total_timesteps = 10000
        
        # Call the function
        model, scaler = train_drl_model(file_path, total_timesteps)
        
        # Assertions
        mock_load_and_preprocess_data.assert_called_once_with(file_path)
        mock_dummy_vec_env.assert_called_once()
        mock_ppo.assert_called_once_with('MlpPolicy', mock_env_instance, verbose=1)
        mock_model_instance.learn.assert_called_once_with(total_timesteps=total_timesteps)
        mock_model_instance.save.assert_called_once_with("ppo_trading_model")
        
        self.assertEqual(model, mock_model_instance)
        self.assertEqual(scaler, mock_scaler)

if __name__ == '__main__':
    unittest.main()
