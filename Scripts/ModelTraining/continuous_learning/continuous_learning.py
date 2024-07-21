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
        # Attempt to load data from DataStore
        data = self.data_store.load_data(self.ticker)
        if data is None:
            logger.info(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}")
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            self.data_store.save_data(data, self.ticker, processed=False)
        else:
            logger.info(f"Data loaded from DataStore: {self.data_store.csv_dir}/{self.ticker}_data.csv")

        if 'Close' not in data.columns:
            logger.error("The 'Close' column is missing in the data.")
            raise ValueError("The 'Close' column is missing in the data.")

        return data

    def train_model(self, total_timesteps=10000):
        logger.info(f"Training model for {self.ticker} with {total_timesteps} timesteps")
        env = make_vec_env(lambda: self.TradingEnv(self.data, self.transaction_cost), n_envs=1)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        self.model = model
        logger.info(f"Model trained and saved to {self.model_path}")

    def backtest_model(self):
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

        logger.info(f"Backtesting completed with Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
        return total_reward, final_balance, mfe, mae

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

    class TradingEnv(gym.Env):
        metadata = {'render.modes': ['human']}

        def __init__(self, data, transaction_cost=0.001):
            super().__init__()
            if isinstance(data, tuple):
                raise ValueError("Data should be a DataFrame, not a tuple.")
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
            reward = current_value - self.initial_balance
            return reward

        def _get_observation(self):
            return np.array([self.balance, self.shares_held, self.price])

        def render(self, mode='human', close=False):
            profit = self.balance + (self.shares_held * self.price) - self.initial_balance
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares Held: {self.shares_held}')
            print(f'Price: {self.price}')
            print(f'Profit: {profit}')
