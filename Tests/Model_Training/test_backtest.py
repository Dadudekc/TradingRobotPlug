import unittest
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

print("Project Root:", project_root)
print("Current Sys Path:", sys.path)

import yfinance as yf
from Scripts.Data_Fetchers.trading_functions import fetch_stock_data, backtest_drl_model, TradingEnv, plot_backtest_results

class TestTradingFunctions(unittest.TestCase):
    
    def setUp(self):
        # Set up any initial data or state
        self.ticker = 'AAPL'
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.model_path = 'path/to/drl_model.zip'  # Ensure you have a valid path
        self.transaction_cost = 0.001
        
        # Fetch stock data
        self.data = fetch_stock_data(self.ticker, self.start_date, self.end_date)
        
        # Simulate a trained model
        self.env = make_vec_env(lambda: TradingEnv(self.data), n_envs=1)
        self.model = PPO('MlpPolicy', self.env)
        self.model.save(self.model_path)

    def test_fetch_stock_data(self):
        data = fetch_stock_data(self.ticker, self.start_date, self.end_date)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_backtest_drl_model(self):
        total_reward, final_balance, mfe, mae = backtest_drl_model(self.data, self.model_path, self.transaction_cost)
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(final_balance, float)
        self.assertIsInstance(mfe, float)
        self.assertIsInstance(mae, float)

    def test_trading_env_reset(self):
        env = TradingEnv(self.data)
        obs = env.reset()
        self.assertEqual(len(obs), 3)
        self.assertEqual(env.balance, env.initial_balance)
        self.assertEqual(env.shares_held, 0)
        self.assertEqual(env.total_reward, 0)

    def test_trading_env_step(self):
        env = TradingEnv(self.data)
        obs = env.reset()
        action = [0, 0.5]  # Example action: Buy with 50% of balance
        obs, reward, done, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn('balance', info)
        self.assertIn('price', info)
    
    def tearDown(self):
        # Clean up after tests
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
