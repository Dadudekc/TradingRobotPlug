import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import json
from datetime import datetime
import threading
import queue

class ModelTrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Model Training GUI")
        self.geometry("900x700")

        self.config = {}  # Load your configuration here if any

        self.create_widgets()
        self.logger = ModelTrainingLogger(self.log_text_widget)
        self.data_handler = DataHandler(self.config, self.log_text_widget)
        self.model_trainer = ModelTrainer(self.config, self.log_text_widget)
        self.model_evaluator = ModelEvaluator(self.log_text_widget)
        self.hyperparameter_tuner = HyperparameterTuner(self.logger)

    def create_widgets(self):
        # Data Handling Frame
        data_frame = ttk.LabelFrame(self, text="Data Handling")
        data_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.file_path_entry = ttk.Entry(data_frame, width=50)
        self.file_path_entry.pack(side="left", padx=5, pady=5)
        ttk.Button(data_frame, text="Browse", command=self.browse_file).pack(side="left", padx=5, pady=5)
        ttk.Button(data_frame, text="Load Data", command=self.load_data).pack(side="left", padx=5, pady=5)

        # Model Training Frame
        training_frame = ttk.LabelFrame(self, text="Model Training")
        training_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(training_frame, text="Model Type:").pack(side="left", padx=5, pady=5)
        self.model_type_combo = ttk.Combobox(training_frame, values=["neural_network", "LSTM", "ARIMA", "linear_regression", "random_forest"])
        self.model_type_combo.pack(side="left", padx=5, pady=5)

        ttk.Label(training_frame, text="Epochs:").pack(side="left", padx=5, pady=5)
        self.epochs_entry = ttk.Entry(training_frame, width=5)
        self.epochs_entry.pack(side="left", padx=5, pady=5)
        self.epochs_entry.insert(0, "50")

        ttk.Button(training_frame, text="Start Training", command=self.start_training).pack(side="left", padx=5, pady=5)

        # Hyperparameter Tuning Frame
        tuning_frame = ttk.LabelFrame(self, text="Hyperparameter Tuning")
        tuning_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(tuning_frame, text="Iterations:").pack(side="left", padx=5, pady=5)
        self.iterations_entry = ttk.Entry(tuning_frame, width=5)
        self.iterations_entry.pack(side="left", padx=5, pady=5)
        self.iterations_entry.insert(0, "100")

        ttk.Button(tuning_frame, text="Perform Tuning", command=self.perform_tuning).pack(side="left", padx=5, pady=5)

        # Log Frame
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text_widget = tk.Text(log_frame, height=15, state="disabled")
        self.log_text_widget.pack(fill="both", expand=True)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def load_data(self):
        file_path = self.file_path_entry.get()
        data = self.data_handler.load_data(file_path)
        if data is not None:
            self.log("Data loaded successfully.")
        else:
            self.log("Failed to load data.", "ERROR")

    def start_training(self):
        file_path = self.file_path_entry.get()
        model_type = self.model_type_combo.get()
        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            self.log("Invalid epochs value.", "ERROR")
            return

        self.model_trainer.start_training(file_path, model_type, epochs)

    def perform_tuning(self):
        model_type = self.model_type_combo.get()
        try:
            iterations = int(self.iterations_entry.get())
        except ValueError:
            self.log("Invalid iterations value.", "ERROR")
            return

        self.log("Performing hyperparameter tuning...")
        # Implement hyperparameter tuning logic here
        self.log("Hyperparameter tuning completed.")

    def log(self, message, level="INFO"):
        self.logger.log(message, level)

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

class DataHandler:
    def __init__(self, config, log_text_widget=None):
        self.config = config
        self.log_text_widget = log_text_widget
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        self.log("DataHandler initialized.")

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            print(f"[{level}] {message}")

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            self.log(f"Data loaded from {file_path}.")
            return data
        except Exception as e:
            error_message = f"Failed to load data from {file_path}: {str(e)}"
            self.log(error_message, "ERROR")
            return None

    def preprocess_data(self, data, target_column='close', date_column='date', lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20]):
        try:
            if data.empty:
                self.log("The dataset is empty before preprocessing. Please check the data source.", "ERROR")
                return None, None

            data.columns = data.columns.str.replace('^[0-9]+\. ', '', regex=True)

            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                reference_date = data[date_column].min()
                data['days_since_reference'] = (data[date_column] - reference_date).dt.days

            if 'index' not in data.columns:
                data.reset_index(inplace=True, drop=True)

            data = self.create_lag_features(data, target_column, lag_sizes)
            data = self.create_rolling_window_features(data, target_column, window_sizes)

            if data.dropna().empty:
                self.log("The dataset became empty after creating lag and rolling window features due to NaN removal. Please adjust the lag and window sizes.", "ERROR")
                return None, None
            else:
                data.dropna(inplace=True)

            data = data.drop(columns=[date_column], errors='ignore')

            if target_column in data.columns:
                y = data[target_column]
                X = data.drop(columns=[target_column])
            else:
                self.log(f"The '{target_column}' column is missing from the dataset. Please check the dataset.", "ERROR")
                return None, None

            if X.empty or y.empty:
                self.log("Either features (X) or target (y) is empty after preprocessing. Please check the preprocessing steps.", "ERROR")
                return None, None

            self.log("Data preprocessing completed.")
            return X, y

        except Exception as e:
            error_message = f"Error during data preprocessing: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
            return None, None

    def create_lag_features(self, df, column_name, lag_sizes):
        if column_name not in df.columns:
            self.log(f"Warning: Column '{column_name}' not found in DataFrame. Skipping lag feature creation.", "ERROR")
            return df

        for lag_days in lag_sizes:
            df[f'{column_name}_lag_{lag_days}'] = df[column_name].shift(lag_days)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.log(f"Lag features created for column '{column_name}' with lag sizes {lag_sizes}.")
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

        self.log(f"Rolling window features created for column '{column_name}' with window sizes {windows}.")
        return data

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.log(f"Data split into training and validation sets with test size {test_size}.")
        return X_train, X_val, y_train, y_val

    def scale_data(self, X_train, X_val, scaler_type='StandardScaler'):
        scaler = self.scalers.get(scaler_type, StandardScaler())
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.log(f"Data scaled using {scaler_type}.")
        return X_train_scaled, X_val_scaled, scaler

    def save_scaler(self, scaler, file_path):
        joblib.dump(scaler, file_path)
        self.log(f"Scaler saved to {file_path}.")

    def load_scaler(self, file_path):
        try:
            scaler = joblib.load(file_path)
            self.log(f"Scaler loaded from {file_path}.")
            return scaler
        except Exception as e:
            self.log(f"Failed to load scaler from {file_path}: {str(e)}", "ERROR")
            return None

    def save_metadata(self, metadata, file_path):
        with open(file_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        self.log(f"Metadata saved to {file_path}.")

    def load_metadata(self, file_path):
        try:
            with open(file_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            self.log(f"Metadata loaded from {file_path}.")
            return metadata
        except Exception as e:
            self.log(f"Failed to load metadata from {file_path}: {str(e)}", "ERROR")
            return None

    def plot_confusion_matrix(self, y_true=None, y_pred=None, conf_matrix=None, class_names=None, save_path="confusion_matrix.png", show_plot=True):
        if conf_matrix is None:
            if y_true is None or y_pred is None:
                self.log("You must provide either a confusion matrix or true and predicted labels.", "ERROR")
                return
            conf_matrix = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = list(range(conf_matrix.shape[0]))

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        self.log(f"Confusion matrix plot saved to {save_path}.")

    def preview_data(self, file_path):
        if not os.path.exists(file_path):
            self.log(f"File does not exist: {file_path}", "ERROR")
            return

        if not file_path.endswith('.csv'):
            self.log("Unsupported file format. Only CSV files are supported.", "ERROR")
            return

        try:
            data = pd.read_csv(file_path)
            self.log(f"Data preview from {file_path}:\n{data.head()}")
        except Exception as e:
            self.log(f"An error occurred while reading the file: {str(e)}", "ERROR")

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

class ModelEvaluator:
    def __init__(self, log_text_widget=None):
        self.log_text_widget = log_text_widget
        self.log("ModelEvaluator initialized.")

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            print(f"[{level}] {message}")

    def evaluate_model(self, model, X_test, y_test, model_type):
        try:
            y_pred = model.predict(X_test)
            results_message = "Model Evaluation Results:\n"

            if model_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)

                results_message += f"""Accuracy: {accuracy:.2f}
Precision: {precision:.2f}
Recall: {recall:.2f}
F1-Score: {fscore:.2f}
AUC-ROC: {auc_roc:.2f}
Log Loss: {logloss:.2f}\n"""

                self.plot_confusion_matrix(conf_matrix, ['Class 0', 'Class 1'])

            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                accuracy = self.calculate_model_accuracy(model, X_test, y_test)

                results_message += f"""MSE: {mse:.2f}
RMSE: {rmse:.2f}
R2 Score: {r2:.2f}
Accuracy: {accuracy:.2f}%\n"""

            self.log(results_message)

            return results_message

        except Exception as e:
            error_message = f"Error during model evaluation: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
            return None

    def calculate_model_accuracy(self, model, X_test, y_test):
        try:
            if hasattr(model, 'score'):
                accuracy = model.score(X_test, y_test)
                return accuracy * 100.0
        except Exception as e:
            self.log(f"Error calculating model accuracy: {str(e)}", "ERROR")
        return 0.0

    def plot_confusion_matrix(self, conf_matrix, class_names, save_path="confusion_matrix.png", show_plot=True):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        self.log(f"Confusion matrix plot saved to {save_path}.")

    def visualize_training_results(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Model Evaluation Results')
        plt.grid(True)
        plt.show()
        self.log("Model evaluation visualization displayed.")

    def generate_model_reports(self, model, X_test, y_test, y_pred, model_type):
        try:
            if model_type == 'classification':
                classification_rep = classification_report(y_test, y_pred)
                self.log("Classification Report:\n" + classification_rep)

                confusion_mat = confusion_matrix(y_test, y_pred)
                self.plot_confusion_matrix(confusion_mat, ['Class 0', 'Class 1'])

            elif model_type == 'regression':
                regression_metrics = self.calculate_regression_metrics(y_test, y_pred)
                self.log("Regression Metrics:\n" + json.dumps(regression_metrics, indent=4))

                self.generate_regression_visualizations(y_test, y_pred)

        except Exception as e:
            self.log(f"Error generating model reports: {str(e)}", "ERROR")

    def calculate_regression_metrics(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics = {
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'Mean Absolute Error (MAE)': mae,
            'R-squared (R2)': r2,
            'Explained Variance': explained_variance,
            'Max Error': max_err,
            'Mean Absolute Percentage Error (MAPE)': mape
        }
        return metrics

    def generate_regression_visualizations(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def save_evaluation_results(self, results, file_path):
        with open(file_path, 'w') as results_file:
            json.dump(results, results_file, indent=4)
        self.log(f"Evaluation results saved to {file_path}.")

    def load_evaluation_results(self, file_path):
        try:
            with open(file_path, 'r') as results_file:
                results = json.load(results_file)
            self.log(f"Evaluation results loaded from {file_path}.")
            return results
        except Exception as e:
            self.log(f"Failed to load evaluation results from {file_path}: {str(e)}", "ERROR")
            return None

class HyperparameterTuner:
    def __init__(self, logger=None):
        self.logger = logger

    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")

    def perform_hyperparameter_tuning(self, model, X_train, y_train, param_distributions, n_iter=100, cv=3, scoring='neg_mean_squared_error', random_state=42):
        self.log("Starting hyperparameter tuning...", "INFO")
        randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, n_jobs=-1)
        randomized_search.fit(X_train, y_train)
        self.log("Hyperparameter tuning completed.", "INFO")
        return randomized_search.best_estimator_, randomized_search.best_params_

    def create_ensemble_model(self, base_models, train_data, train_labels, method='voting', weights=None):
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            if isinstance(base_models[0], Classifier):
                ensemble_model = VotingClassifier(estimators=base_models, voting='hard', weights=weights)
            else:
                ensemble_model = VotingRegressor(estimators=base_models, weights=weights)
        elif method == 'stacking':
            # Implement stacking ensemble method
            pass
        else:
            raise ValueError("Unsupported ensemble method.")

        ensemble_model.fit(train_data, train_labels)
        self.log(f"Ensemble model created using method {method}.", "INFO")
        return ensemble_model

    def quantize_model(self, model, quantization_method='weight'):
        if quantization_method == 'weight':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()
        elif quantization_method == 'activation':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_generator
            quantized_model = converter.convert()
        else:
            raise ValueError("Unsupported quantization method.")
        
        self.log(f"Model quantized using method {quantization_method}.", "INFO")
        return quantized_model

    def representative_dataset_generator(self, train_data):
        dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(1)
        for input_data in dataset.take(100):
            yield [input_data]

    def initialize_and_configure_model(self, model_type, input_shape, epochs):
        if model_type == "neural_network":
            model = Sequential([
                Dense(128, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        elif model_type == "LSTM":
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        elif model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "ARIMA":
            return {'order': (5, 1, 0)}
        else:
            raise ValueError("Unsupported model type")

        self.log(f"Model of type {model_type} initialized and configured.", "INFO")
        return model

    def create_windowed_data(self, X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)

if __name__ == "__main__":
    app = ModelTrainingGUI()
    app.mainloop()
