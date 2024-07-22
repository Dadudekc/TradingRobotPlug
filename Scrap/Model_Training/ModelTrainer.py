from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import threading
import traceback
import json
from datetime import datetime
from tkinter import filedialog

class ModelTrainer:
    def __init__(self, config, log_text_widget):
        self.config = config
        self.logger = ModelTrainingLogger(log_text_widget)
        self.queue = queue.Queue()
        self.is_debug_mode = False
        self.trained_model = None
        self.trained_scaler = None
        self.model_configs = {
            "neural_network": {
                "epochs": {"label": "Epochs:", "default": 50},
                "window_size": {"label": "Window Size:", "default": 30}
            },
            "LSTM": {
                "epochs": {"label": "Epochs:", "default": 50},
                "window_size": {"label": "Window Size:", "default": 30}
            },
            "ARIMA": {
                "p_value": {"label": "ARIMA p-value:", "default": 1},
                "d_value": {"label": "ARIMA d-value:", "default": 1},
                "q_value": {"label": "ARIMA q-value:", "default": 1}
            },
            "linear_regression": {
                "regularization": {"label": "Regularization(alpha):", "default": 0.01}
            },
            "random_forest": {
                "n_estimators": {"label": "Number of Trees (n_estimators):", "default": 100},
                "max_depth": {"label": "Max Depth:", "default": None},
                "min_samples_split": {"label": "Min Samples Split:", "default": 2},
                "min_samples_leaf": {"label": "Min Samples Leaf:", "default": 1}
            }
        }

    def log(self, message, level="INFO"):
        self.logger.log(message, level)

    def preprocess_data_with_feature_engineering(self, data, lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20]):
        if data.empty:
            self.log("The dataset is empty before preprocessing. Please check the data source.", "ERROR")
            return None, None

        data.columns = data.columns.str.replace('^[0-9]+\. ', '', regex=True)

        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            reference_date = data['date'].min()
            data['days_since_reference'] = (data['date'] - reference_date).dt.days

        if 'index' not in data.columns:
            data.reset_index(inplace=True, drop=True)

        data = self.create_lag_features(data, 'close', lag_sizes)
        data = self.create_rolling_window_features(data, 'close', window_sizes)

        if data.dropna().empty:
            self.log("The dataset became empty after creating lag and rolling window features due to NaN removal. Please adjust the lag and window sizes.", "ERROR")
            return None, None
        else:
            data.dropna(inplace=True)

        data = data.drop(columns=['date'], errors='ignore')

        if 'close' in data.columns:
            y = data['close']
            X = data.drop(columns=['close'])
        else:
            self.log("The 'close' column is missing from the dataset. Please check the dataset.", "ERROR")
            return None, None

        if X.empty or y.empty:
            self.log("Either features (X) or target (y) is empty after preprocessing. Please check the preprocessing steps.", "ERROR")
            return None, None

        return X, y

    def create_lag_features(self, df, column_name, lag_sizes):
        if column_name not in df.columns:
            self.log(f"Warning: Column '{column_name}' not found in DataFrame. Skipping lag feature creation.", "ERROR")
            return df

        for lag_days in lag_sizes:
            df[f'{column_name}_lag_{lag_days}'] = df[column_name].shift(lag_days)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df

    def create_rolling_window_features(self, data, column_name, windows, method='pad'):
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()

            if method == 'interpolate':
                data[f'{column_name}_rolling_mean_{window}'].interpolate(method='linear', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].interpolate(method='linear', inplace=True)
            elif method == 'pad':
                data[f'{column_name}_rolling_mean_{window}'].fillna(method='pad', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].fillna(method='pad', inplace=True)
            else:
                data.fillna(data.mean(), inplace=True)

        return data

    def start_training(self, data_file_path, model_type, epochs=50):
        try:
            self.disable_training_button()
            self.log("Training started...", "INFO")

            data = pd.read_csv(data_file_path)
            self.log("Data loading and preprocessing started.", "INFO")
            X, y = self.preprocess_data_with_feature_engineering(data)

            if X is None or y is None or X.empty or y.empty:
                self.log("Preprocessing resulted in empty data. Aborting training.", "ERROR")
                return

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            trained_model = None
            if model_type in ['neural_network', 'LSTM']:
                trained_model, _ = self.train_neural_network_or_lstm_with_regularization_and_transfer_learning(X_train, y_train, X_val, y_val, model_type, epochs)
            elif model_type == 'linear_regression':
                trained_model = self.train_linear_regression_with_auto_optimization(X_train, y_train, X_val, y_val)
            elif model_type == 'random_forest':
                trained_model = self.train_random_forest_with_auto_optimization(X_train, y_train, X_val, y_val)
            elif model_type == "ARIMA":
                self.train_arima_model_in_background(y_train)

            self.log("Training completed successfully.", "INFO")
            self.save_trained_model(trained_model, model_type)

        except Exception as e:
            error_message = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
        finally:
            self.enable_training_button()

    def train_neural_network_or_lstm_with_regularization_and_transfer_learning(self, X_train, y_train, X_val, y_val, model_type, epochs=100, pretrained_model_path=None, previous_model_metrics=None):
        if pretrained_model_path:
            model = self.load_model(pretrained_model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
        else:
            model = Sequential()

        if model_type == "neural_network":
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            X_train_reshaped, X_val_reshaped = X_train, X_val
        elif model_type == "LSTM":
            model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1) if not isinstance(X_train, np.ndarray) else X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1) if not isinstance(X_val, np.ndarray) else X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])

        y_pred_val = model.predict(X_val_reshaped).flatten()

        mse = mean_squared_error(y_val, y_pred_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        r2 = r2_score(y_val, y_pred_val)
        self.log(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return model, (mse, rmse, r2)

    def train_linear_regression_with_auto_optimization(self, X_train, y_train, X_val, y_val):
        param_grid = {'alpha': np.logspace(-4, 0, 50)}
        model = Ridge()
        randomized_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', verbose=2)
        randomized_search.fit(X_train, y_train)

        best_model = randomized_search.best_estimator()
        r2 = best_model.score(X_val, y_val)
        y_pred_val = best_model.predict(X_val)
        mse, rmse = mean_squared_error(y_val, y_pred_val), mean_squared_error(y_val, y_pred_val, squared=False)
        self.log(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", "INFO")

        return best_model

    def train_random_forest_with_auto_optimization(self, X_train, y_train, X_val, y_val, random_state=None):
        param_grid = {
            'n_estimators': np.linspace(10, 300, num=20, dtype=int),
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }

        rf = RandomForestRegressor(random_state=random_state)
        rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=3, verbose=1, random_state=random_state, n_jobs=-1)
        rf_random_search.fit(X_train, y_train)

        best_rf_model = rf_random_search.best_estimator()
        r2 = best_rf_model.score(X_val, y_val)
        y_pred_val = best_rf_model.predict(X_val)
        mse, rmse = mean_squared_error(y_val, y_pred_val), mean_squared_error(y_val, y_pred_val, squared=False)
        self.log(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return best_rf_model

    def train_arima_model_in_background(self, close_prices, threshold=100):
        def background_training(close_prices):
            results = {'predictions': [], 'errors': [], 'parameters': {'order': (5, 1, 0)}, 'performance_metrics': {}}
            train_size = int(len(close_prices) * 0.8)
            train, test = close_prices[:train_size], close_prices[train_size:]
            history = list(train)

            for t in range(len(test)):
                try:
                    model = ARIMA(history, order=results['parameters']['order'])
                    model_fit = model.fit()
                    forecast = model_fit.forecast()[0]
                    results['predictions'].append(forecast)
                    obs = test[t]
                    history.append(obs)
                except Exception as e:
                    self.log(f"Error training ARIMA model at step {t}: {e}", level="ERROR")
                    results['errors'].append(str(e))

            mse = mean_squared_error(test, results['predictions'])
            self.log(f"Test MSE: {mse:.2f}")

            if mse < threshold:
                self.log("Your ARIMA model seems promising for forecasting stock prices.", level="INFO")
            else:
                self.log("Consider different ARIMA parameters or models for better forecasting accuracy.", level="WARNING")

            if mse < threshold:
                self.log("Your ARIMA model performs well! Consider using the same or similar parameters (p, d, q) for similar datasets.", level="INFO")
            else:
                self.log("Consider trying different combinations of (p, d, q) parameters. AIC and BIC from the model summary can guide the selection.", level="INFO")

            self.log("Tip: A lower AIC or BIC value usually indicates a better model fit. Use these metrics to compare different ARIMA configurations.", level="INFO")

        threading.Thread(target=background_training, args=(close_prices,), daemon=True).start()
        self.log("ARIMA model training started in background...", level="INFO")

    def save_trained_model(self, model=None, model_type=None, scaler=None, file_path=None):
        model = model or self.trained_model
        scaler = scaler or self.trained_scaler

        if model is None or model_type is None:
            self.log("No trained model available to save or model type not provided.", "ERROR")
            return

        file_extension = ".joblib"

        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"{model_type}_{timestamp}{file_extension}"
            file_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                                    initialfile=default_filename,
                                                    filetypes=[(f"{model_type.upper()} Files", f"*{file_extension}"), ("All Files", "*.*")])

            if not file_path:
                self.log("Save operation canceled by user.", "INFO")
                return

        if not file_path.endswith(file_extension):
            file_path += "_model" + file_extension

        joblib.dump(model, file_path)
        self.log(f"Model of type '{model_type}' saved successfully at {file_path}")

        if scaler is not None:
            scaler_file_path = file_path.replace(file_extension, "_scaler.joblib")
            joblib.dump(scaler, scaler_file_path)
            self.log(f"Scaler saved successfully at {scaler_file_path}")

        metadata = self.construct_metadata(model, model_type, scaler)
        metadata_file_path = file_path.replace(file_extension, "_metadata.json")
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        self.log(f"Metadata saved to {metadata_file_path}")

    def construct_metadata(self, model, model_type, scaler=None):
        metadata = {
            'model_type': model_type,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        if hasattr(model, 'get_params'):
            metadata['model_parameters'] = {param: str(value) for param, value in model.get_params().items()}

        if hasattr(model, 'named_steps'):
            metadata['pipeline_steps'] = {}
            for name, step in model.named_steps.items():
                step_representation = {
                    'class': step.__class__.__name__,
                    'parameters': {param: str(value) for param, value in step.get_params().items()}
                }
                metadata['pipeline_steps'][name] = step_representation

        if scaler:
            metadata['scaler'] = {
                'class': scaler.__class__.__name__,
                'parameters': {param: str(value) for param, value in scaler.get_params().items()}
            }

        return metadata

    def disable_training_button(self):
        # Implement the logic to disable the training button in your GUI
        pass

    def enable_training_button(self):
        # Implement the logic to enable the training button in your GUI
        pass

class ModelTrainingLogger:
    def __init__(self, log_text_widget):
        self.log_text_widget = log_text_widget
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def log(self, message, level="INFO"):
        self.logger.log(getattr(logging, level), message)
        self.log_text_widget.config(state='normal')
        self.log_text_widget.insert('end', f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        self.log_text_widget.config(state='disabled')
        self.log_text_widget.see('end')

# Example usage:
# config = ...  # Load your configuration here
# log_text_widget = ...  # Your Tkinter Text widget for logging
# trainer = ModelTrainer(config, log_text_widget)
# trainer.start_training('data/file/path.csv', 'neural_network', 50)
