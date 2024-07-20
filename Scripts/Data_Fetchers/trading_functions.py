import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gym
from gym import spaces


class StockTradingModel:
    def __init__(self, ticker, start_date, end_date, model_path='ppo_trading_model', transaction_cost=0.001):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_path = model_path
        self.transaction_cost = transaction_cost
        self.data = self.fetch_stock_data()

    def fetch_stock_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data

    def train_model(self, total_timesteps=10000):
        env = make_vec_env(lambda: self.TradingEnv(self.data, self.transaction_cost), n_envs=1)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        self.model = model

    def backtest_model(self):
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
