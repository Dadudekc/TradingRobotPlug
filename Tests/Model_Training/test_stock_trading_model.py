import unittest
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stock_trading_model import StockTradingModel  # Ensure this matches your module name

class TestStockTradingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This will run once before all tests
        cls.ticker = "AAPL"
        cls.start_date = "2020-01-01"
        cls.end_date = "2020-12-31"
        cls.model_path = "ppo_trading_model"
        cls.transaction_cost = 0.001
        cls.model = StockTradingModel(cls.ticker, cls.start_date, cls.end_date, cls.model_path, cls.transaction_cost)

    def test_fetch_stock_data(self):
        data = self.model.fetch_stock_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_train_model(self):
        self.model.train_model(total_timesteps=1000)
        self.assertTrue(os.path.exists(self.model.model_path))

    def test_backtest_model(self):
        total_reward, final_balance, mfe, mae = self.model.backtest_model()
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(final_balance, float)
        self.assertIsInstance(mfe, float)
        self.assertIsInstance(mae, float)

    def test_plot_backtest_results(self):
        # Simulate some data for plotting
        step_rewards = np.cumsum(np.random.randn(100))
        step_prices = np.random.randn(100).cumsum() + 100
        self.model.plot_backtest_results(step_rewards, step_prices)
        # If no exceptions, we assume the plot was successful

    @classmethod
    def tearDownClass(cls):
        # This will run once after all tests
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

if __name__ == "__main__":
    unittest.main()
