import unittest
import os
import pandas as pd
from stable_baselines3 import PPO
from train_drl_model import DRLModelTrainer  # Ensure this matches your module name

class TestDRLModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This will run once before all tests
        cls.file_path = "path_to_your_data.csv"
        cls.total_timesteps = 1000  # Set lower timesteps for faster tests
        cls.transaction_cost = 0.001
        cls.trainer = DRLModelTrainer(cls.file_path, cls.total_timesteps, cls.transaction_cost)

    def test_preprocess_data(self):
        X_train, X_val, y_train, y_val = self.trainer.preprocess_data()
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_val, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_val, pd.Series)
        self.assertFalse(X_train.empty)
        self.assertFalse(X_val.empty)
        self.assertFalse(y_train.empty)
        self.assertFalse(y_val.empty)

    def test_create_env(self):
        X_train, X_val, y_train, y_val = self.trainer.preprocess_data()
        env = self.trainer.create_env(X_train)
        self.assertIsNotNone(env)
    
    def test_train_model(self):
        model, scaler_path = self.trainer.train_model()
        self.assertIsInstance(model, PPO)
        self.assertTrue(os.path.exists(self.trainer.model_path))
        self.assertTrue(os.path.exists(scaler_path))

    @classmethod
    def tearDownClass(cls):
        # This will run once after all tests
        if os.path.exists(cls.trainer.model_path):
            os.remove(cls.trainer.model_path)
        if os.path.exists(cls.trainer.scaler_path):
            os.remove(cls.trainer.scaler_path)

if __name__ == "__main__":
    unittest.main()
