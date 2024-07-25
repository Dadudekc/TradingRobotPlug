# models/arima_model_trainer.py

import threading
import logging
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from datetime import datetime
import traceback
from sklearn.preprocessing import StandardScaler
import numpy as np

class ARIMAModelTrainer:
    def __init__(self, close_prices, threshold=100):
        self.close_prices = close_prices
        self.threshold = threshold
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('ARIMA_Test')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

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

        history = list(scaled_train)

        # Automatically find the best ARIMA parameters
        self.display_message("Finding the best ARIMA parameters...", "INFO")
        try:
            model = pm.auto_arima(history, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, seasonal=False,
                                trace=True, error_action='ignore', suppress_warnings=True, maxiter=1000)
            results['parameters']['order'] = model.order
            self.display_message(f"Selected ARIMA parameters: {model.order}", "INFO")
        except Exception as e:
            self.display_message(f"Error finding ARIMA parameters: {e}", "ERROR")
            return

        # Fit the model on the initial history
        model_fit = pm.ARIMA(order=model.order, maxiter=1000)
        model_fit.fit(history)

        scaled_test = pd.Series(scaled_test).reset_index(drop=True)  # Reset index of test Series

        for t in range(len(scaled_test)):
            try:
                self.display_message(f"Training step {t}/{len(scaled_test)}", "DEBUG")
                forecast = model_fit.predict(n_periods=1)[0]  # Predict the next period
                self.display_message(f"Forecast at step {t}: {forecast}", "DEBUG")
                results['predictions'].append(forecast)
                obs = scaled_test[t]
                history.append(obs)
                # Update the model with the new observation
                model_fit.update([obs])
            except ValueError as ve:
                self.display_message(f"ValueError at step {t}: {ve}", "ERROR")
                results['errors'].append(str(ve))
            except IndexError as ie:
                self.display_message(f"IndexError at step {t}: {ie}", "ERROR")
                results['errors'].append(str(ie))
            except Exception as e:
                self.display_message(f"Unexpected error at step {t}: {e}", "ERROR")
                results['errors'].append(str(e))
                # Log the traceback for the unexpected error
                self.display_message(f"Traceback: {traceback.format_exc()}", "DEBUG")
                break  # Exit the loop if an unexpected error occurs

        if len(results['predictions']) > 0:
            # Inverse scale predictions and test data
            predictions = scaler.inverse_transform(np.array(results['predictions']).reshape(-1, 1)).flatten()
            test_actual = scaler.inverse_transform(scaled_test[:len(results['predictions'])].values.reshape(-1, 1)).flatten()

            mse = mean_squared_error(test_actual, predictions)
            self.display_message(f"Test MSE: {mse:.2f}", "INFO")

            if mse < self.threshold:
                self.display_message("Your ARIMA model seems promising for forecasting stock prices.", "INFO")
            else:
                self.display_message("Consider different ARIMA parameters or models for better forecasting accuracy.", "WARNING")
        else:
            self.display_message("No valid predictions were made by the ARIMA model. Please check the model configuration and data.", "ERROR")

        # Log final performance metrics
        results['performance_metrics']['mse'] = mse
        self.display_message(f"Final performance metrics: MSE = {mse:.2f}", "INFO")

        # Save the results
        pd.DataFrame({'Actual': test_actual, 'Predicted': predictions}).to_csv('arima_predictions.csv', index=False)
        self.display_message("Results saved to arima_predictions.csv", "INFO")



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
