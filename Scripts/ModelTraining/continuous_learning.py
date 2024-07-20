import time
import pandas as pd
from stable_baselines3 import PPO
from train_drl_model import train_drl_model
from Scripts.Data_Fetchers.trading_functions import backtest_drl_model
from DataHandler import DataHandler

class ContinuousLearning:
    def __init__(self, file_path, retrain_interval=86400, total_timesteps=10000, config=None):
        if config is None:
            config = {}
        self.file_path = file_path
        self.retrain_interval = retrain_interval
        self.total_timesteps = total_timesteps
        self.data_handler = DataHandler(config)
        
    def run(self):
        while True:
            print("Training model...")
            model, scaler_path = train_drl_model(self.file_path, self.total_timesteps)
            
            if model is None or scaler_path is None:
                print("Model training failed. Retrying...")
                continue
            
            print("Backtesting model...")
            total_reward, final_balance, mfe, mae = backtest_drl_model(self.file_path, "ppo_trading_model")
            
            print(f"Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
            
            time.sleep(self.retrain_interval)

if __name__ == "__main__":
    learning_process = ContinuousLearning('path_to_your_data.csv')
    learning_process.run()
