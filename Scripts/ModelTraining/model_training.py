import os
import threading
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import logging

# Setup logger
logger = logging.getLogger('ModelTraining')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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

    def preprocess_data_with_feature_engineering(self, data):
        """Preprocess the data by creating lag and rolling window features."""
        if data.empty:
            self.display_message("The dataset is empty before preprocessing. Please check the data source.", "ERROR")
            return None, None, None, None

        data.columns = data.columns.str.replace('^[0-9]+\\. ', '', regex=True)

        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            reference_date = data['date'].min()
            data['days_since_reference'] = (data['date'] - reference_date).dt.days

        data = self.create_lag_features(data, 'close', [1, 2, 3, 5, 10])
        data = self.create_rolling_window_features(data, 'close', [5, 10, 20])
        
        if data.empty:
            self.display_message("Preprocessing resulted in empty data. Please check the data transformations.", "ERROR")
            return None, None, None, None

        data = data.dropna()  # Drop any rows with NaN values

        if 'close' in data.columns:
            y = data['close']
            X = data.drop(columns=['close', 'date'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            self.display_message("The 'close' column is missing from the dataset. Please check the dataset.", "ERROR")
            return None, None, None, None

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

    def disable_training_button(self):
        """Disable the training button to prevent multiple clicks."""
        try:
            if hasattr(self, 'training_button') and self.training_button is not None:
                self.training_button.config(state=tk.DISABLED)
                self.display_message("Training button disabled to prevent multiple clicks.", "INFO")
            else:
                self.display_message("Training button not found or already disabled.", "WARNING")
        except Exception as e:
            self.display_message(f"Error disabling training button: {str(e)}", "ERROR")

    def enable_training_button(self):
        """Enable the training button."""
        try:
            if hasattr(self, 'training_button') and self.training_button is not None:
                self.training_button.config(state=tk.NORMAL)
                self.display_message("Training button enabled.", "INFO")
            else:
                self.display_message("Training button not found or already enabled.", "WARNING")
        except Exception as e:
            self.display_message(f"Error enabling training button: {str(e)}", "ERROR")

    def create_lag_features(self, df, column_name, lag_sizes):
        """Create lag features for the specified column."""
        for lag in lag_sizes:
            df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
        df.fillna(method='ffill', inplace=True)
        return df

    def create_rolling_window_features(self, data, column_name, windows):
        """Create rolling window features for the specified column."""
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()
            data.fillna(method='ffill', inplace=True)
        return data

    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """Train a linear regression model with hyperparameter tuning."""
        param_grid = {'alpha': np.logspace(-4, 0, 50)}
        model = Ridge()
        randomized_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', verbose=2)
        
        try:
            randomized_search.fit(X_train, y_train)
        except Exception as e:
            self.display_message(f"Error during model training: {str(e)}", "ERROR")
            return None
        
        self.display_message("Randomized Search Results:", "INFO")
        results_df = pd.DataFrame(randomized_search.cv_results_)
        results_str = results_df[['param_alpha', 'mean_test_score', 'std_test_score']].to_string()
        self.display_message(results_str, "INFO")

        cv_scores = cross_val_score(randomized_search.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores_str = ", ".join([f"{score:.2f}" for score in cv_scores])
        self.display_message(f"CV Scores: {cv_scores_str}", "INFO")

        best_model = randomized_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        r2 = best_model.score(X_val, y_val)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", "INFO")
        
        best_alpha = randomized_search.best_params_['alpha']
        self.display_message(f"Best regularization strength (alpha): {best_alpha:.4f}. Consider using this as a starting point for your next training session.", "INFO")

        return best_model

    def train_random_forest(self, X_train, y_train, X_val, y_val, random_state=None):
        """Train a random forest model with hyperparameter tuning."""
        param_grid = {
            'n_estimators': np.linspace(10, 300, num=20, dtype=int),
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }
        
        rf = RandomForestRegressor(random_state=random_state)
        rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=3, verbose=1, random_state=random_state, n_jobs=-1)
        rf_random_search.fit(X_train, y_train)
        
        best_rf_model = rf_random_search.best_estimator_
        r2 = best_rf_model.score(X_val, y_val)
        y_pred_val = best_rf_model.predict(X_val)
        mse, rmse = mean_squared_error(y_val, y_pred_val), mean_squared_error(y_val, y_pred_val, squared=False)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", "INFO")
        
        best_params = rf_random_search.best_params_
        self.display_message(f"Best parameters found: {best_params}. Use these parameters as a baseline for your next training session.", "INFO")

        return best_rf_model

    def train_neural_network_or_lstm(self, X_train, y_train, X_val, y_val, model_type, epochs=100, pretrained_model_path=None):
        """Train a neural network or LSTM model based on the specified model type."""
        if pretrained_model_path:
            model = load_model(pretrained_model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
        else:
            model = Sequential()

        if model_type == "neural_network":
            for layer in self.model_configs['neural_network']['layers']:
                if layer['type'] == 'dense':
                    model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer']))
                elif layer['type'] == 'batch_norm':
                    model.add(BatchNormalization())
                elif layer['type'] == 'dropout':
                    model.add(Dropout(rate=layer['rate']))
            model.compile(optimizer=self.model_configs['neural_network']['optimizer'], loss=self.model_configs['neural_network']['loss'])
            X_train_reshaped, X_val_reshaped = X_train, X_val
        elif model_type == "LSTM":
            model.add(LSTM(units=self.model_configs['LSTM']['layers'][0]['units'], return_sequences=self.model_configs['LSTM']['layers'][0]['return_sequences'], kernel_regularizer=self.model_configs['LSTM']['layers'][0]['kernel_regularizer'], input_shape=(X_train.shape[1], 1)))
            for layer in self.model_configs['LSTM']['layers'][1:]:
                if layer['type'] == 'batch_norm':
                    model.add(BatchNormalization())
                elif layer['type'] == 'dropout':
                    model.add(Dropout(rate=layer['rate']))
                elif layer['type'] == 'dense':
                    model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer']))
            model.compile(optimizer=self.model_configs['LSTM']['optimizer'], loss=self.model_configs['LSTM']['loss'])
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])

        y_pred_val = model.predict(X_val_reshaped).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", "INFO")

        return model

    def train_arima_model(self, close_prices, threshold=100):
        """Train an ARIMA model in the background."""
        def background_training(close_prices):
            self.display_message("Starting ARIMA training background process...", "INFO")
            results = {'predictions': [], 'errors': [], 'parameters': {}, 'performance_metrics': {}}
            train_size = int(len(close_prices) * 0.8)
            train, test = close_prices[:train_size], close_prices[train_size:]
            history = list(train)

            self.display_message(f"Train size: {train_size}, Test size: {len(test)}", "INFO")
            self.display_message(f"Train data: {train.head()}", "DEBUG")
            self.display_message(f"Test data: {test.head()}", "DEBUG")

            # Automatically find the best ARIMA parameters
            self.display_message("Finding the best ARIMA parameters...", "INFO")
            try:
                model = pm.auto_arima(history, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
                results['parameters']['order'] = model.order
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
                    results['predictions'].append(forecast)
                    obs = test[t]
                    history.append(obs)
                except Exception as e:
                    self.display_message(f"Error training ARIMA model at step {t}: {e}", "ERROR")
                    results['errors'].append(str(e))
            
            if len(results['predictions']) > 0:
                mse = mean_squared_error(test, results['predictions'])
                self.display_message(f"Test MSE: {mse:.2f}")

                if mse < threshold:
                    self.display_message("Your ARIMA model seems promising for forecasting stock prices.", "INFO")
                else:
                    self.display_message("Consider different ARIMA parameters or models for better forecasting accuracy.", "WARNING")
            else:
                self.display_message("No valid predictions were made by the ARIMA model. Please check the model configuration and data.", "ERROR")

            # Log final performance metrics
            results['performance_metrics']['mse'] = mse
            self.display_message(f"Final performance metrics: MSE = {mse:.2f}", "INFO")

        if close_prices is None or close_prices.empty:
            self.display_message("The provided close_prices data is empty or None.", "ERROR")
            return

        self.display_message("Initiating ARIMA model training...", "INFO")
        training_thread = threading.Thread(target=background_training, args=(close_prices,), daemon=True)
        training_thread.start()
        self.display_message("ARIMA model training started in background...", "INFO")

        # Wait for the background thread to complete
        training_thread.join()
        self.display_message("ARIMA model training background process completed.", "INFO")

    def async_evaluate_model(self, model, X_test, y_test, model_type):
        """Asynchronously evaluate the model."""
        if model_type == 'regression':
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {'mse': mse, 'r2': r2}
        return {}

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

    def save_model(self, model, model_type):
        """Save the trained model to a file."""
        model_dir = "models"
        if not os.path.exists(model_dir):
            self.display_message(f"Creating directory {model_dir}", "INFO")
            os.makedirs(model_dir)
        else:
            self.display_message(f"Directory {model_dir} already exists", "INFO")
            
        model_path = f"{model_dir}/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.display_message(f"Saving model to {model_path}", "INFO")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f'Saving model to {model_path}')
        return model_path

    def start_training(self, X_train, y_train, X_val, y_val, model_type, epochs=50):
        """Start the training process."""
        try:
            self.display_message("Training started...", "INFO")

            trained_model = None
            if model_type in ['neural_network', 'LSTM']:
                trained_model = self.train_neural_network_or_lstm(X_train, y_train, X_val, y_val, model_type, epochs)
            elif model_type == 'linear_regression':
                trained_model = self.train_linear_regression(X_train, y_train, X_val, y_val)
            elif model_type == 'random_forest':
                trained_model = self.train_random_forest(X_train, y_train, X_val, y_val)
            elif model_type == "ARIMA":
                self.train_arima_model(y_train)

            self.display_message("Training completed successfully.", "INFO")
            self.save_model(trained_model, model_type)
            return trained_model

        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            self.display_message(error_message, "ERROR")
            return None
