import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model

class LSTMModelTrainer:
    def __init__(self, logger=None, model_save_path="best_model.keras", scaler_save_path="scaler.pkl"):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model_save_path = Path(model_save_path)  # Path for saving the model
        self.scaler_save_path = Path(scaler_save_path)  # Path for saving the scaler

    def preprocess_data(self, X_train, X_val):
        """Preprocess data by handling missing values and scaling."""
        self.logger.info("Preprocessing data...")
        self.logger.debug(f"Initial X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        joblib.dump(scaler, self.scaler_save_path)  # Save the scaler

        self.logger.debug(f"Scaled X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}")
        return X_train_scaled, X_val_scaled

    def train_lstm(self, X_train, y_train, X_val, y_val, model, epochs=100):
        """Train an LSTM model."""
        self.logger.info("Starting LSTM model training...")
        try:
            X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
            checkpoint = ModelCheckpoint(self.model_save_path, save_best_only=True, monitor='val_loss', mode='min')
            lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20.0))

            self.logger.info(f"X_train_scaled shape: {X_train_scaled.shape}")
            self.logger.info(f"y_train shape: {y_train.shape}")
            self.logger.info(f"X_val_scaled shape: {X_val_scaled.shape}")
            self.logger.info(f"y_val shape: {y_val.shape}")

            model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=epochs, batch_size=32,
                      callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler])

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
        """Evaluate the model's performance on the test dataset."""
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
        if len(data) <= time_steps or len(target) <= time_steps:
            raise ValueError(f"Data length ({len(data)}) or target length ({len(target)}) is too short for the specified time_steps ({time_steps}).")

        xs, ys = [], []
        for i in range(len(data) - time_steps):
            x = data[i:(i + time_steps)]
            try:
                y = target[i + time_steps]  # Ensure correct indexing
            except KeyError as e:
                print(f"KeyError encountered at index {i + time_steps}: {e}")
                continue  # Skip this iteration if the index is not found
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"X_val shape: {X_val.shape}")
        
        model_config = {
            'layers': [
                {'type': 'bidirectional_lstm', 'units': trial.suggest_int('units_lstm', 50, 200), 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
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
