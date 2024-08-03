import logging
import os
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import joblib

# Adjust import path based on your project structure
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # Assuming project root is three levels up

# Add the 'Utilities' directory to sys.path
utilities_dir = project_root / 'Scripts' / 'Utilities'
sys.path.append(str(utilities_dir))

# Debug print to confirm the path
print("Corrected Project root path:", project_root)
print("Adding Utilities directory to sys.path:", utilities_dir)

from model_training_utils import LoggerHandler, DataLoader, DataPreprocessor
# Now import the DataStore class
try:
    from data_store import DataStore
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Configure logging
logger_handler = LoggerHandler()
logger = logger_handler.logger

class AdvancedLSTMModelTrainer:
    def __init__(self, logger=None, model_save_path="best_model.keras", scaler_save_path="scaler.pkl"):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model_save_path = Path(model_save_path)
        self.scaler_save_path = Path(scaler_save_path)

    def preprocess_data(self, X_train, X_val):
        self.logger.log("Preprocessing data...", level="INFO")
        self.logger.log(f"Initial X_train shape: {X_train.shape}, X_val shape: {X_val.shape}", level="DEBUG")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        joblib.dump(scaler, self.scaler_save_path)

        self.logger.log(f"Scaled X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}", level="DEBUG")
        return X_train_scaled, X_val_scaled


    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, callbacks=None):
        self.logger.log("Starting LSTM model training...", level="INFO")
        try:
            X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val)

            if callbacks is None:
                callbacks = []

            input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
            model = self.build_lstm_model(input_shape)

            self.logger.log(f"X_train_scaled shape: {X_train_scaled.shape}", level="INFO")
            self.logger.log(f"y_train shape: {y_train.shape}", level="INFO")
            self.logger.log(f"X_val_scaled shape: {X_val_scaled.shape}", level="INFO")
            self.logger.log(f"y_val shape: {y_val.shape}", level="INFO")

            model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=epochs, batch_size=32,
                    callbacks=callbacks)

            y_pred_val = model.predict(X_val_scaled).flatten()

            self.logger.log(f"Predicted y_val shape: {y_pred_val.shape}", level="INFO")
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.log(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", level="INFO")

            model.save(self.model_save_path)
            return model
        except Exception as e:
            self.logger.log(f"Error occurred during LSTM model training: {e}", level="ERROR")
            raise



    def evaluate_model(self, X_test, y_test):
        self.logger.info("Evaluating model on test data...")
        try:
            if X_test.size == 0 or y_test.size == 0:
                raise ValueError("Test data is empty. Cannot evaluate model.")
            
            model = load_model(self.model_save_path)
            scaler = joblib.load(self.scaler_save_path)

            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            y_pred_test = model.predict(X_test_scaled).flatten()

            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)

            self.logger.info(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")

    @staticmethod
    def create_sequences(data, target, time_steps=10):
        xs, ys = [], []
        for i in range(len(data) - time_steps):
            x = data[i:(i + time_steps)]
            try:
                y = target.iloc[i + time_steps]
                xs.append(x)
                ys.append(y)
            except IndexError as e:
                logger.error(f"IndexError at position {i}: {e}")
        return np.array(xs), np.array(ys)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"X_val shape: {X_val.shape}")
        
        model_config = {
            'layers': [
                {'type': 'bidirectional_lstm', 'units': trial.suggest_int('units_lstm', 50, 200), 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                {'type': 'attention'},
                {'type': 'batch_norm'},
                {'type': 'dropout', 'rate': trial.suggest_float('dropout_rate', 0.2, 0.5)},
                {'type': 'dense', 'units': trial.suggest_int('units_dense', 10, 50), 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)}
            ],
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta']),
            'loss': 'mean_squared_error'
        }

        model = LSTMModelConfig.lstm_model((X_train.shape[1], X_train.shape[2]), model_config)
        model = self.train_lstm(X_train, y_train, X_val, y_val, model, epochs=50)

        if model is None:
            raise optuna.exceptions.TrialPruned()
        
        y_pred_val = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        return mse

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

def main():
    data_file_path = 'C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv'
    model_save_path = 'best_model.keras'
    scaler_save_path = 'scaler.pkl'

    # Initialize LoggerHandler with the logger
    logger_handler = LoggerHandler(logger=logger)
    
    # Pass the LoggerHandler instance to DataLoader, not just logger
    data_loader = DataLoader(logger_handler)
    
    config_manager = None
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load data using DataLoader
    data = data_loader.load_data(data_file_path)

    if data is not None:
        X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(data)

        if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
            time_steps = 10

            logger_handler.log(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
            logger_handler.log(f"X_val length: {len(X_val)}, y_val length: {len(y_val)}")

            trainer = AdvancedLSTMModelTrainer(logger_handler, model_save_path, scaler_save_path)
            try:
                logger_handler.log(f"Creating sequences with time_steps: {time_steps}")
                X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

                logger_handler.log(f"X_train_seq shape: {X_train_seq.shape}")
                logger_handler.log(f"y_train_seq shape: {y_train_seq.shape}")
                logger_handler.log(f"X_val_seq shape: {X_val_seq.shape}")
                logger_handler.log(f"y_val_seq shape: {y_val_seq.shape}")

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                trained_model = trainer.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50,
                    callbacks=[early_stopping, reduce_lr]
                )

                if trained_model:
                    trainer.evaluate_model(X_val_seq, y_val_seq)
            except ValueError as e:
                logger_handler.log(f"ValueError in creating sequences: {e}", "ERROR")
            except KeyError as e:
                logger_handler.log(f"KeyError encountered: {e}", "ERROR")
        else:
            logger_handler.log("Data preprocessing failed.", "ERROR")
    else:
        logger_handler.log("Data loading failed.", "ERROR")

if __name__ == "__main__":
    main()
