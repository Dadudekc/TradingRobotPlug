import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import joblib
from DataHandler import DataHandler
from Scripts.Data_Fetchers.trading_functions import TradingEnv

class DRLModelTrainer:
    def __init__(self, file_path, total_timesteps=10000, transaction_cost=0.001, config=None):
        if config is None:
            config = {}
        self.file_path = file_path
        self.total_timesteps = total_timesteps
        self.transaction_cost = transaction_cost
        self.data_handler = DataHandler(config)
        self.model_path = "ppo_trading_model"
        self.scaler_path = "scaler.pkl"

    def preprocess_data(self):
        X_train, X_val, y_train, y_val = self.data_handler.preprocess_data(self.file_path)
        return X_train, X_val, y_train, y_val

    def create_env(self, X_train):
        env = make_vec_env(lambda: TradingEnv(X_train, self.transaction_cost), n_envs=1)
        return env

    def train_model(self):
        X_train, X_val, y_train, y_val = self.preprocess_data()
        
        if X_train is None or X_val is None or y_train is None or y_val is None:
            print("Data preprocessing failed.")
            return None, None
        
        env = self.create_env(X_train)
        
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=self.total_timesteps)
        model.save(self.model_path)
        
        joblib.dump(self.data_handler.scalers['StandardScaler'], self.scaler_path)
        
        return model, self.scaler_path

if __name__ == "__main__":
    trainer = DRLModelTrainer('path_to_your_data.csv')
    model, scaler = trainer.train_model()
