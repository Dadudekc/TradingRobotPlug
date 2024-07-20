import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import yfinance as yf
from trading_robot import fetch_stock_data, backtest_drl_model, plot_backtest_results, train_and_backtest_model, train_model

class TestTradingRobot(unittest.TestCase):

    @patch('trading_robot.yf.download')
    def test_fetch_stock_data(self, mock_download):
        # Setup
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
            'Open': np.random.rand(5),
            'High': np.random.rand(5),
            'Low': np.random.rand(5),
            'Close': np.random.rand(5),
            'Volume': np.random.randint(1, 100, size=5)
        })
        mock_download.return_value = mock_data

        # Execute
        data = fetch_stock_data('AAPL', '2022-01-01', '2022-01-05')

        # Verify
        mock_download.assert_called_once_with('AAPL', start='2022-01-01', end='2022-01-05')
        pd.testing.assert_frame_equal(data, mock_data)

    @patch('trading_robot.PPO.load')
    @patch('trading_robot.make_vec_env')
    def test_backtest_drl_model(self, mock_make_vec_env, mock_ppo_load):
        # Setup
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model
        mock_model.predict.return_value = (np.array([0]), None)
        mock_env.reset.return_value = np.array([0])
        mock_env.step.return_value = (np.array([0]), [1], False, [{'balance': 1000, 'price': 100}])

        data = pd.DataFrame({'Close': np.random.rand(100)})
        model_path = 'test_model.zip'

        # Execute
        total_reward, final_balance, mfe, mae, rewards, prices = backtest_drl_model(data, model_path)

        # Verify
        mock_make_vec_env.assert_called_once()
        mock_ppo_load.assert_called_once_with(model_path)
        self.assertEqual(total_reward, 1)
        self.assertEqual(final_balance, 1000)
        self.assertIsInstance(rewards, list)
        self.assertIsInstance(prices, list)

    @patch('trading_robot.plt.show')
    def test_plot_backtest_results(self, mock_plt_show):
        # Setup
        step_rewards = np.cumsum(np.random.randn(100))
        step_prices = np.random.randn(100).cumsum() + 100

        # Execute
        plot_backtest_results(step_rewards, step_prices)

        # Verify
        mock_plt_show.assert_called_once()

    @patch('trading_robot.fetch_stock_data')
    @patch('trading_robot.PPO')
    @patch('trading_robot.make_vec_env')
    @patch('trading_robot.plot_backtest_results')
    def test_train_and_backtest_model(self, mock_plot_backtest_results, mock_make_vec_env, mock_ppo, mock_fetch_stock_data):
        # Setup
        mock_fetch_stock_data.return_value = pd.DataFrame({'Close': np.random.rand(100)})
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        mock_model.predict.return_value = (np.array([0]), None)
        mock_env.reset.return_value = np.array([0])
        mock_env.step.return_value = (np.array([0]), [1], False, [{'balance': 1000, 'price': 100}])

        # Execute
        results = train_and_backtest_model('AAPL', '2022-01-01', '2022-12-31', 'test_model.zip')

        # Verify
        mock_fetch_stock_data.assert_called_once()
        mock_make_vec_env.assert_called_once()
        mock_ppo.assert_called_once()
        mock_plot_backtest_results.assert_called_once()
        self.assertIn('total_reward', results)
        self.assertIn('final_balance', results)
        self.assertIn('mfe', results)
        self.assertIn('mae', results)

    @patch('trading_robot.fetch_stock_data')
    @patch('trading_robot.PPO')
    @patch('trading_robot.make_vec_env')
    def test_train_model(self, mock_make_vec_env, mock_ppo, mock_fetch_stock_data):
        # Setup
        mock_fetch_stock_data.return_value = pd.DataFrame({'Close': np.random.rand(100)})
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        mock_model.learn.return_value = None
        mock_model.save.return_value = None

        # Execute
        train_model('AAPL', '2022-01-01', '2022-12-31', 'test_model.zip')

        # Verify
        mock_fetch_stock_data.assert_called_once()
        mock_make_vec_env.assert_called_once()
        mock_ppo.assert_called_once()
        mock_model.learn.assert_called_once()
        mock_model.save.assert_called_once_with('test_model.zip')

if __name__ == "__main__":
    unittest.main()
