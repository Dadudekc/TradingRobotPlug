# File: arima_model_trainer.py
# Location: Scripts/Training/
# Description: Script for training ARIMA models on stock data with background processing and logging.

import os
import sys
import pandas as pd
import numpy as np
import threading
from datetime import datetime
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import traceback

# Set up project root dynamically
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # Adjust based on your project structure
sys.path.append(str(project_root / 'Scripts' / 'Utilities'))

# Set up relative paths for resources and logs
resources_path = project_root / 'resources'
log_path = project_root / 'logs'

# Ensure the directories exist
resources_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
import logging
log_file = log_path / 'application.log'
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from model_training_utils import setup_logger, get_project_root, DataLoader, DataPreprocessor, ConfigManager, LoggerHandler

try:
    from data_store import DataStore
except ModuleNotFoundError as e:
    logging.error(f"Error importing modules: {e}")
    logging.error(f"sys.path: {sys.path}")
    sys.exit(1)

class ARIMAModelTrainer:
    def __init__(self, symbol, threshold=100):
        self.symbol = symbol
        self.threshold = threshold
        self.store = DataStore()
        project_root = get_project_root()
        log_file = project_root / f'logs/arima_{symbol}.log'
        self.logger = setup_logger(f'ARIMA_{self.symbol}', log_file)
        self.close_prices = None

    def load_data(self):
        """Load data for the symbol using DataStore."""
        df = self.store.load_data(self.symbol)
        if df is None or df.empty:
            self.logger.error(f"Failed to load data for {self.symbol}")
            raise ValueError(f"No data available for {self.symbol}")
        return df['close']

    def display_message(self, message, level="INFO"):
        """Log messages with timestamps."""
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.debug(message)

    def make_stationary(self, data):
        """Make the series stationary by differencing and performing ADF test."""
        result = adfuller(data)
        if result[1] <= 0.05:
            self.display_message("Data is already stationary.", "INFO")
            return data
        else:
            self.display_message("Data is not stationary, differencing will be applied.", "INFO")
            return data.diff().dropna()

    def background_training(self):
        self.display_message("Starting ARIMA training background process...", "INFO")
        results = {'predictions': [], 'errors': [], 'parameters': {}, 'performance_metrics': {}}
        train_size = int(len(self.close_prices) * 0.8)
        train, test = self.close_prices[:train_size], self.close_prices[train_size:]
        history = list(train)

        self.display_message(f"Train size: {train_size}, Test size: {len(test)}", "INFO")
        self.display_message(f"Train data: {train.head()}", "DEBUG")
        self.display_message(f"Test data: {test.head()}", "DEBUG")

        # Check if the test data is of the correct numeric type
        if not (test.dtype in [np.float64, np.float32]):
            test = pd.to_numeric(test, errors='coerce')

        # Scale the data
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
        scaled_test = scaler.transform(test.values.reshape(-1, 1)).flatten()

        # Make the data stationary
        scaled_train = self.make_stationary(pd.Series(scaled_train)).values
        scaled_test = np.array(pd.Series(scaled_test).reset_index(drop=True))

        history = list(scaled_train)

        # Manually set ARIMA parameters with increased maxiter
        self.display_message("Testing manually set ARIMA parameters with increased maxiter...", "INFO")
        try:
            model = pm.ARIMA(order=(1, 1, 1), solver='lbfgs', maxiter=1000)  # Increased maxiter
            model.fit(history)
            results['parameters']['order'] = model.order
            self.display_message(f"Selected ARIMA parameters: {model.order}", "INFO")

            if hasattr(model, 'arparams_'):
                self.display_message(f"AR Coefficients: {model.arparams_}")
            else:
                self.display_message("AR Coefficients are not available.", "WARNING")

            if hasattr(model, 'maparams_'):
                self.display_message(f"MA Coefficients: {model.maparams_}")
            else:
                self.display_message("MA Coefficients are not available.", "WARNING")

        except Exception as e:
            self.display_message(f"Error fitting ARIMA model: {e}", "ERROR")
            return

        for t in range(len(scaled_test)):
            try:
                self.display_message(f"Training step {t}/{len(scaled_test)}", "DEBUG")
                forecast = model.predict(n_periods=1)[0]  # Predict the next period
                self.display_message(f"Forecast at step {t}: {forecast}", "DEBUG")
                results['predictions'].append(forecast)
                obs = scaled_test[t]
                history.append(obs)
                model.update([obs])
            except ValueError as ve:
                self.display_message(f"ValueError at step {t}: {ve}", "ERROR")
                results['errors'].append(str(ve))
            except IndexError as ie:
                self.display_message(f"IndexError at step {t}: {ie}", "ERROR")
                results['errors'].append(str(ie))
            except Exception as e:
                self.display_message(f"Unexpected error at step {t}: {e}", "ERROR")
                results['errors'].append(str(e))
                self.display_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
                break

        if len(results['predictions']) > 0:
            predictions = scaler.inverse_transform(np.array(results['predictions']).reshape(-1, 1)).flatten()
            test_actual = scaler.inverse_transform(scaled_test[:len(results['predictions'])].reshape(-1, 1)).flatten()

            mse = mean_squared_error(test_actual, predictions)
            self.display_message(f"Test MSE: {mse:.2f}", "INFO")

            if mse < self.threshold:
                self.display_message(f"Your ARIMA model for {self.symbol} seems promising for forecasting stock prices.", "INFO")
            else:
                self.display_message(f"Consider different ARIMA parameters or models for {self.symbol} for better forecasting accuracy.", "WARNING")
        else:
            self.display_message(f"No valid predictions were made by the ARIMA model for {self.symbol}. Please check the model configuration and data.", "ERROR")

        results['performance_metrics']['mse'] = mse
        self.display_message(f"Final performance metrics: MSE = {mse:.2f}", "INFO")

        output_file = project_root / f'arima_predictions_{self.symbol}.csv'
        pd.DataFrame({'Actual': test_actual, 'Predicted': predictions}).to_csv(output_file, index=False)
        self.display_message(f"Results for {self.symbol} saved to {output_file}", "INFO")

    def train(self, data=None):
        """Main method to start ARIMA training."""
        if data is not None:
            self.close_prices = data
        else:
            self.close_prices = self.load_data()

        if self.close_prices is None or self.close_prices.empty:
            self.display_message("The provided close_prices data is empty or None.", "ERROR")
            return

        self.display_message("Initiating ARIMA model training...", "INFO")
        training_thread = threading.Thread(target=self.background_training, daemon=True)
        training_thread.start()
        self.display_message("ARIMA model training started in background...", "INFO")

        # Wait for the background thread to complete
        training_thread.join()
        self.display_message("ARIMA model training background process completed.", "INFO")


# Example of how to use these classes:
if __name__ == "__main__":
    trainer = ARIMAModelTrainer(symbol="AAPL")
    trainer.train()
