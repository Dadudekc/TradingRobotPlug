import sys
import os
from keras.regularizers import l1_l2
from keras.optimizers import Adam

# Add project root to the PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.ModelTraining.model_training.models.neural_network import train_neural_network
from Scripts.ModelTraining.model_training.models.lstm_trainer import train_lstm
from Scripts.ModelTraining.model_training.models.linear_regression import train_linear_regression
from Scripts.ModelTraining.model_training.models.random_forest import RandomForestModel  # Updated import
from Scripts.ModelTraining.model_training.models.arima_model import ARIMAModelTrainer

import threading
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

class ModelTraining:
    def __init__(self, logger):
        self.logger = logger
        self.model_configs = self.initialize_model_configs()

    def initialize_model_configs(self):
        """Initialize model configurations for different model types."""
        return {
            'neural_network': {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                    {'type': 'batch_norm'},
                    {'type': 'dropout', 'rate': 0.2},
                    {'type': 'dense', 'units': 64, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                    {'type': 'dropout', 'rate': 0.2},
                    {'type': 'dense', 'units': 1, 'activation': None, 'kernel_regularizer': None}
                ],
                'optimizer': Adam(learning_rate=1e-4),
                'loss': 'mean_squared_error',
                'epochs': 100
            },
            'LSTM': {
                'layers': [
                    {'type': 'lstm', 'units': 50, 'return_sequences': False, 'input_shape': (None, 1), 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                    {'type': 'batch_norm'},
                    {'type': 'dropout', 'rate': 0.2},
                    {'type': 'dense', 'units': 64, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                    {'type': 'dense', 'units': 1, 'activation': None, 'kernel_regularizer': None}
                ],
                'optimizer': Adam(learning_rate=1e-4),
                'loss': 'mean_squared_error',
                'epochs': 100
            }
        }

    def display_message(self, message, level="INFO"):
        """Log messages with timestamps."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {message}"
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.debug(message)

    def start_training(self, X_train, y_train, X_val, y_val, model_type, epochs=50):
        """Start the training process."""
        try:
            self.display_message("Training started...", "INFO")

            trained_model = None
            if model_type == 'neural_network':
                trained_model = train_neural_network(X_train, y_train, X_val, y_val, self.model_configs['neural_network'], epochs, logger=self.logger)
            elif model_type == 'LSTM':
                trained_model = train_lstm(X_train, y_train, X_val, y_val, self.model_configs['LSTM'], epochs, logger=self.logger)
            elif model_type == 'linear_regression':
                trained_model = train_linear_regression(X_train, y_train, X_val, y_val, logger=self.logger)
            elif model_type == 'random_forest':
                rf_model = RandomForestModel(logger=self.logger)
                trained_model, best_params, mse, rmse, mae, mape, r2 = rf_model.train(X_train, y_train, random_state=42)
                self.display_message(f"Best parameters: {best_params}", "INFO")
                self.display_message(f"Validation MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R²: {r2}", "INFO")
            elif model_type == 'ARIMA':
                arima_trainer = ARIMAModelTrainer(y_train, self.logger)
                arima_trainer.train()

            self.display_message("Training completed successfully.", "INFO")
            if trained_model:
                self.save_model(trained_model, model_type)
            return trained_model

        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            self.display_message(error_message, "ERROR")
            return None

    def save_model(self, model, model_type):
        """Save the trained model to a file."""
        model_dir = os.path.join(project_root, "models")
        if not os.path.exists(model_dir):
            self.display_message(f"Creating directory {model_dir}", "INFO")
            os.makedirs(model_dir)
        else:
            self.display_message(f"Directory {model_dir} already exists", "INFO")
            
        model_path = os.path.join(model_dir, f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        self.display_message(f"Saving model to {model_path}", "INFO")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path

    def visualize_training_results(self, y_test, y_pred):
        """Visualize the training results."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, edgecolor='k', alpha=0.7)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.grid(True)
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='blue')
        plt.xlabel('Residuals')
        plt.title('Residuals Distribution')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Replace with actual data
    X_train, y_train = np.random.rand(100, 10), np.random.rand(100)
    X_val, y_val = np.random.rand(20, 10), np.random.rand(20)

    training_instance = ModelTraining(logger=logger)
    model = training_instance.start_training(X_train, y_train, X_val, y_val, model_type='random_forest', epochs=50)
