import sys
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2
import optuna

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]  # Adjust this according to the correct level
    sys.path.append(str(project_root))

from Scripts.Utilities.DataHandler import DataHandler
from Scripts.ModelTraining.model_training.models.lstm.lstm_config import LSTMModelConfig

# Set up logging
log_dir = Path("C:/TheTradingRobotPlug/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "lstm_model_trainer.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger('AdvancedLSTMModelTrainer')

class AdvancedLSTMModelTrainer:
    def __init__(self, logger=None, model_save_path="best_model.keras", scaler_save_path="scaler.pkl"):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model_save_path = Path(model_save_path)
        self.scaler_save_path = Path(scaler_save_path)

    def preprocess_data(self, X_train, X_val):
        self.logger.info("Preprocessing data...")
        self.logger.debug(f"Initial X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        joblib.dump(scaler, self.scaler_save_path)

        self.logger.debug(f"Scaled X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}")
        return X_train_scaled, X_val_scaled

    def train_lstm(self, X_train, y_train, X_val, y_val, model, epochs=100, callbacks=None):
        self.logger.info("Starting LSTM model training...")
        try:
            X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val)

            if callbacks is None:
                callbacks = []

            self.logger.info(f"X_train_scaled shape: {X_train_scaled.shape}")
            self.logger.info(f"y_train shape: {y_train.shape}")
            self.logger.info(f"X_val_scaled shape: {X_val_scaled.shape}")
            self.logger.info(f"y_val shape: {y_val.shape}")

            model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=epochs, batch_size=32,
                      callbacks=callbacks)

            y_pred_val = model.predict(X_val_scaled).flatten()
            self.logger.info(f"Predicted y_val shape: {y_pred_val.shape}")
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            return model

        except Exception as e:
            self.logger.error(f"Error occurred during LSTM model training: {e}")
            return None

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

    data_handler = DataHandler(logger=logger)
    data = data_handler.load_data(data_file_path)
    
    if data is not None:
        X_train, X_val, y_train, y_val = data_handler.preprocess_data(data)

        if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
            time_steps = 10

            logger.info(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
            logger.info(f"X_val length: {len(X_val)}, y_val length: {len(y_val)}")

            trainer = AdvancedLSTMModelTrainer(logger, model_save_path, scaler_save_path)
            try:
                logger.info(f"Creating sequences with time_steps: {time_steps}")
                X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

                logger.info(f"X_train_seq shape: {X_train_seq.shape}")
                logger.info(f"y_train_seq shape: {y_train_seq.shape}")
                logger.info(f"X_val_seq shape: {X_val_seq.shape}")
                logger.info(f"y_val_seq shape: {y_val_seq.shape}")

                model_params = {
                    'layers': [
                        {'type': 'bidirectional_lstm', 'units': 100, 'return_sequences': True, 'kernel_regularizer': None},
                        {'type': 'attention'},
                        {'type': 'batch_norm'},
                        {'type': 'dropout', 'rate': 0.3},
                        {'type': 'dense', 'units': 50, 'activation': 'relu', 'kernel_regularizer': None}
                    ],
                    'optimizer': 'adam',
                    'loss': 'mean_squared_error'
                }

                model_config = LSTMModelConfig.lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), model_params)
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                trained_model = trainer.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50,
                    callbacks=[early_stopping, reduce_lr]
                )

                if trained_model:
                    trainer.evaluate_model(X_val_seq, y_val_seq)
            except ValueError as e:
                logger.error(f"ValueError in creating sequences: {e}")
            except KeyError as e:
                logger.error(f"KeyError encountered: {e}")
        else:
            logger.error("Data preprocessing failed.")
    else:
        logger.error("Data loading failed.")

if __name__ == "__main__":
    main()
