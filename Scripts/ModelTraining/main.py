import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
from trading_env import TradingEnv
from risk_management import RiskManager
import tkinter as tk
from tkinter import ttk

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def backtest_drl_model(data, model_path, transaction_cost=0.001):
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)
    env = make_vec_env(lambda: TradingEnv(data), n_envs=1)
    model = PPO.load(model_path)
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
    return total_reward, final_balance, mfe, mae, rewards, prices

def plot_backtest_results(step_rewards, step_prices):
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

def train_and_backtest_model(ticker, start_date, end_date, model_save_path, training_timesteps=10000):
    data = fetch_stock_data(ticker, start_date, end_date)
    risk_manager = RiskManager(max_drawdown=0.2, stop_loss=0.05, take_profit=0.1)
    env = make_vec_env(lambda: TradingEnv(data, risk_manager=risk_manager), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=training_timesteps)
    model.save(model_save_path)
    total_reward, final_balance, mfe, mae, rewards, prices = backtest_drl_model(data, model_save_path)
    plot_backtest_results(rewards, prices)
    return {
        'total_reward': total_reward,
        'final_balance': final_balance,
        'mfe': mfe,
        'mae': mae
    }

def train_model(ticker, start_date, end_date, model_save_path, training_timesteps=10000):
    data = fetch_stock_data(ticker, start_date, end_date)
    env = make_vec_env(lambda: TradingEnv(data), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=training_timesteps)
    model.save(model_save_path)
    print("Model training completed and saved.")

def run_reinforcement_learning():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    model_save_path = model_path_entry.get()
    results = train_and_backtest_model(ticker, start_date, end_date, model_save_path)
    results_label.config(text=f"Backtest Results: {results}")

def run_regular_training():
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    model_save_path = model_path_entry.get()
    train_model(ticker, start_date, end_date, model_save_path)
    results_label.config(text="Model training completed and saved.")

app = tk.Tk()
app.title("Trading Robot")

frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Ticker:").grid(row=0, column=0, sticky=tk.W)
ticker_entry = ttk.Entry(frame)
ticker_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W)
start_date_entry = ttk.Entry(frame)
start_date_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W)
end_date_entry = ttk.Entry(frame)
end_date_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Model Save Path:").grid(row=3, column=0, sticky=tk.W)
model_path_entry = ttk.Entry(frame)
model_path_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

frame.columnconfigure(1, weight=1)

button_frame = ttk.Frame(frame, padding="10")
button_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

reinforcement_button = ttk.Button(button_frame, text="Run Reinforcement Learning", command=run_reinforcement_learning)
reinforcement_button.grid(row=0, column=0, padx=5)

regular_training_button = ttk.Button(button_frame, text="Run Regular Training", command=run_regular_training)
regular_training_button.grid(row=0, column=1, padx=5)

results_label = ttk.Label(frame, text="")
results_label.grid(row=5, column=0, columnspan=2, pady=10)

app.mainloop()
