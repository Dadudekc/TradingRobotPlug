import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gym
from gym import spaces
import logging
import sys
import json

# Workaround for OpenMP runtime issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Adjust the Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.config_handling import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockTradingModel:
    def __init__(self, ticker, start_date, end_date, model_path='ppo_trading_model', transaction_cost=0.001, config_file='config.ini'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_path = model_path
        self.transaction_cost = transaction_cost

        # Initialize ConfigManager
        self.config_manager = ConfigManager(config_file)

        # Initialize DataStore
        data_store_config = {
            'csv_dir': self.config_manager.get('DATA_STORE', 'csv_dir', 'C:/TheTradingRobotPlug/data/alpha_vantage'),
            'db_path': self.config_manager.get('DATA_STORE', 'db_path', 'C:/TheTradingRobotPlug/data/trading_data.db')
        }
        self.data_store = DataStore(**data_store_config)

        self.data = self.fetch_stock_data()

    def fetch_stock_data(self):
        try:
            # Attempt to load data from DataStore
            data = self.data_store.load_data(self.ticker)
            if data is None:
                logger.info(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}")
                data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
                self.data_store.save_data(data, self.ticker, processed=False)
            data.columns = [col.capitalize() for col in data.columns]
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    def train_model(self, total_timesteps=10000):
        try:
            logger.info(f"Training model for {self.ticker} with {total_timesteps} timesteps")
            env = make_vec_env(lambda: self.TradingEnv(self.data, self.transaction_cost), n_envs=1)
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=total_timesteps)
            model.save(self.model_path)
            self.model = model
            logger.info(f"Model trained and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def backtest_model(self):
        try:
            logger.info(f"Backtesting model for {self.ticker}")
            data = self.data.copy()
            data['Date'] = pd.to_datetime(data.index)
            data.set_index('Date', inplace=True)

            env = make_vec_env(lambda: self.TradingEnv(data, self.transaction_cost), n_envs=1)
            model = PPO.load(self.model_path)

            obs = env.reset()
            done = False
            total_reward = 0
            final_balance = 0
            prices = []
            rewards = []
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                final_balance = info[0]['balance']
                prices.append(info[0]['price'])
                rewards.append(total_reward)

            prices = np.array(prices)
            mfe = np.max(prices) - prices[0]
            mae = prices[0] - np.min(prices)

            results = {
                "total_reward": total_reward,
                "final_balance": final_balance,
                "mfe": mfe,
                "mae": mae,
                "sharpe_ratio": self.calculate_sharpe_ratio(rewards),
                "sortino_ratio": self.calculate_sortino_ratio(rewards),
                "max_drawdown": self.calculate_max_drawdown(prices),
                "step_rewards": rewards,
                "step_prices": prices.tolist()
            }

            self.save_backtest_results(results)
            self.plot_backtest_results(rewards, prices)

            logger.info(f"Backtesting completed with Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
            return total_reward, final_balance, mfe, mae
        except Exception as e:
            logger.error(f"Error during model backtesting: {e}")
            raise

    def save_backtest_results(self, results):
        results_file = os.path.join(script_dir, f"{self.ticker}_backtest_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Backtest results saved to {results_file}")

    def plot_backtest_results(self, step_rewards, step_prices):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(step_rewards, label='Cumulative Reward')
        plt.title('Backtest Results')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(step_prices, label='Stock Price', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Stock Price')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def calculate_sharpe_ratio(self, rewards, risk_free_rate=0.01):
        returns = np.diff(rewards) / rewards[:-1]
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_sortino_ratio(self, rewards, risk_free_rate=0.01):
        returns = np.diff(rewards) / rewards[:-1]
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def calculate_max_drawdown(self, prices):
        peak = prices[0]
        max_drawdown = 0
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if (drawdown > max_drawdown):
                max_drawdown = drawdown
        return max_drawdown

    class TradingEnv(gym.Env):
        metadata = {'render.modes': ['human']}

        def __init__(self, data, transaction_cost=0.001):
            super().__init__()
            self.data = data
            self.current_step = 0
            self.initial_balance = 10000
            self.balance = self.initial_balance
            self.shares_held = 0
            self.total_reward = 0
            self.done = False
            self.price = self.data['Close'].iloc[self.current_step]
            self.transaction_cost = transaction_cost

            self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=0, high=np.inf, shape=(3,), dtype=np.float32
            )

        def seed(self, seed=None):
            np.random.seed(seed)

        def reset(self):
            self.current_step = 0
            self.balance = self.initial_balance
            self.shares_held = 0
            self.total_reward = 0
            self.done = False
            self.price = self.data['Close'].iloc[self.current_step]
            return self._get_observation()

        def step(self, action):
            self._take_action(action)
            self.current_step += 1
            self.price = self.data['Close'].iloc[self.current_step]
            reward = self._calculate_reward()
            self.total_reward += reward
            self.done = self.current_step >= len(self.data) - 1
            info = {'balance': self.balance, 'price': self.price}
            return self._get_observation(), reward, self.done, info

        def _take_action(self, action):
            action_type = int(action[0])
            amount = action[1]

            if action_type == 0:  # Buy
                total_possible = self.balance // self.price
                shares_bought = total_possible * amount
                cost = shares_bought * self.price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_bought
            elif action_type == 1:  # Sell
                shares_sold = self.shares_held * amount
                self.balance += shares_sold * self.price * (1 - self.transaction_cost)
                self.shares_held -= shares_sold

        def _calculate_reward(self):
            current_value = self.shares_held * self.price + self.balance
            reward = current_value - self.total_reward
            return reward

        def _get_observation(self):
            return np.array([self.balance, self.shares_held, self.price])


if __name__ == "__main__":
    model = StockTradingModel(ticker="AAPL", start_date="2020-01-01", end_date="2020-12-31")
    model.train_model()
    total_reward, final_balance, mfe, mae = model.backtest_model()
    print(f"Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
    # Optionally, you can call model.plot_backtest_results() with the appropriate parameters
