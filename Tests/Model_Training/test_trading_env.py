import unittest
import numpy as np
from trading_env import TradingEnv

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        # Mock data: 10 steps of trading data with 2 features (e.g., 'Close' and another feature)
        self.mock_data = np.array([
            [100, 1],
            [101, 1],
            [102, 1],
            [103, 1],
            [104, 1],
            [105, 1],
            [106, 1],
            [107, 1],
            [108, 1],
            [109, 1],
        ])
        self.env = TradingEnv(data=self.mock_data, initial_balance=1000)

    def test_reset(self):
        observation = self.env.reset()
        self.assertTrue(np.array_equal(observation, self.mock_data[0]))
        self.assertEqual(self.env.balance, 1000)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.current_step, 0)

    def test_step_hold(self):
        self.env.reset()
        observation, reward, done, _ = self.env.step(0)
        self.assertTrue(np.array_equal(observation, self.mock_data[1]))
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_step_buy(self):
        self.env.reset()
        observation, reward, done, _ = self.env.step(1)
        self.assertEqual(self.env.position, 10)
        self.assertEqual(self.env.balance, 0)
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_step_sell(self):
        self.env.reset()
        self.env.step(1)  # Buy
        observation, reward, done, _ = self.env.step(2)  # Sell
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.balance, 1010)  # 10 shares sold at 101 each
        self.assertEqual(reward, 10)  # Profit of 10
        self.assertFalse(done)

    def test_step_end(self):
        self.env.reset()
        for _ in range(len(self.mock_data) - 1):
            observation, reward, done, _ = self.env.step(0)
        self.assertTrue(done)
        self.assertEqual(self.env.current_step, len(self.mock_data) - 1)

    def test_mfe_mae_calculation(self):
        self.env.reset()
        for _ in range(len(self.mock_data) - 1):
            self.env.step(0)
        # Ensure the final reward includes MFE and MAE
        current_price = self.mock_data[-1, 0]
        mfe = (109 - 100) / 100
        mae = (109 - 100) / 100
        expected_reward = mfe - mae
        self.assertAlmostEqual(self.env.step(0)[1], expected_reward)

if __name__ == '__main__':
    unittest.main()
