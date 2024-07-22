# C:\TheTradingRobotPlug\Scripts\ModelTraining\model_training\models\arima_model.py

import threading
import logging
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from datetime import datetime

class ARIMAModelTrainer:
    def __init__(self, close_prices, logger, threshold=100):
        self.close_prices = close_prices
        self.threshold = threshold
        self.logger = logger
        self.results = {'predictions': [], 'errors': [], 'parameters': {}, 'performance_metrics': {}}

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
        train_size = int(len(self.close_prices) * 0.8)
        train, test = self.close_prices[:train_size], self.close_prices[train_size:]
        history = list(train)

        self.display_message(f"Train size: {train_size}, Test size: {len(test)}", "INFO")
        self.display_message(f"Train data: {train.head()}", "DEBUG")
        self.display_message(f"Test data: {test.head()}", "DEBUG")

        # Automatically find the best ARIMA parameters
        self.display_message("Finding the best ARIMA parameters...", "INFO")
        try:
            model = pm.auto_arima(history, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
            self.results['parameters']['order'] = model.order
            self.display_message(f"Selected ARIMA parameters: {model.order}", "INFO")
        except Exception as e:
            self.display_message(f"Error finding ARIMA parameters: {e}", "ERROR")
            return

        for t in range(len(test)):
            try:
                self.display_message(f"Training step {t}/{len(test)}", "DEBUG")
                model_fit = model.fit(history)
                forecast = model_fit.predict(n_periods=1)[0]
                self.display_message(f"Forecast at step {t}: {forecast}", "DEBUG")
                self.results['predictions'].append(forecast)
                obs = test[t]
                history.append(obs)
            except Exception as e:
                self.display_message(f"Error training ARIMA model at step {t}: {e}", "ERROR")
                self.results['errors'].append(str(e))
        
        if len(self.results['predictions']) > 0:
            mse = mean_squared_error(test, self.results['predictions'])
            self.display_message(f"Test MSE: {mse:.2f}")

            if mse < self.threshold:
                self.display_message("Your ARIMA model seems promising for forecasting stock prices.", "INFO")
            else:
                self.display_message("Consider different ARIMA parameters or models for better forecasting accuracy.", "WARNING")
        else:
            self.display_message("No valid predictions were made by the ARIMA model. Please check the model configuration and data.", "ERROR")

        # Log final performance metrics
        self.results['performance_metrics']['mse'] = mse
        self.display_message(f"Final performance metrics: MSE = {mse:.2f}", "INFO")

        # Save the results
        pd.DataFrame({'Actual': test, 'Predicted': self.results['predictions']}).to_csv('arima_predictions.csv', index=False)
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
