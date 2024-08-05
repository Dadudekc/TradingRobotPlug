from Scripts.Utilities.test2 import setup_logger, load_config, get_project_root
import pandas as pd
import numpy as np
import logging
import threading
from datetime import datetime
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import yaml
import os
import sys


# Adjust import path based on your project structure
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # Assuming project root is three levels up

# Add the 'Utilities' directory to sys.path
utilities_dir = project_root / 'Scripts' / 'Utilities'
sys.path.append(str(utilities_dir))

# Debug print to confirm the path
print("Corrected Project root path:", project_root)
print("Adding Utilities directory to sys.path:", utilities_dir)

# Now import the DataStore class
try:
    from data_store import DataStore
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

class ARIMAModelTrainer:
    def __init__(self, symbol, threshold=100):
        self.symbol = symbol
        self.threshold = threshold
        self.store = DataStore()
        project_root = get_project_root()
        log_file = project_root / f'logs/arima_{symbol}.log'
        self.logger = setup_logger(f'ARIMA_{self.symbol}', log_file)
        self.close_prices = self.load_data()

    def load_data(self):
        """Load data for the symbol using DataStore."""
        df = self.store.load_data(self.symbol)
        if df is None or df.empty:
            self.logger.error(f"Failed to load data for {self.symbol}")
            raise ValueError(f"No data available for {self.symbol}")
        return df['close']

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

        # Scale the data
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
        scaled_test = scaler.transform(test.values.reshape(-1, 1)).flatten()

        # Make the data stationary
        scaled_train = self.make_stationary(pd.Series(scaled_train)).values
        scaled_test = pd.Series(scaled_test).reset_index(drop=True)  # Reset index of test Series

        history = list(scaled_train)

        # Manually set ARIMA parameters with increased maxiter
        self.display_message("Testing manually set ARIMA parameters with increased maxiter...", "INFO")
        try:
            model = pm.ARIMA(order=(1, 1, 1), solver='lbfgs', maxiter=1000)  # Increased maxiter
            model.fit(history)
            results['parameters']['order'] = model.order
            self.display_message(f"Selected ARIMA parameters: {model.order}", "INFO")
            
            # Check if the model attributes exist before accessing
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
                # Update the model with the new observation
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
            test_actual = scaler.inverse_transform(scaled_test[:len(results['predictions'])].values.reshape(-1, 1)).flatten()

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

        pd.DataFrame({'Actual': test_actual, 'Predicted': predictions}).to_csv(f'arima_predictions_{self.symbol}.csv', index=False)
        self.display_message(f"Results for {self.symbol} saved to arima_predictions_{self.symbol}.csv", "INFO")

    def train(self):
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


