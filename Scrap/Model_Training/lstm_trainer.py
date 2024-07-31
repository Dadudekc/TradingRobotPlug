import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Layer

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent.parent
    sys.path.append(str(project_root))
    
    # Ensure the module path is added correctly
    model_training_path = project_root / "Scripts/ModelTraining/model_training/models"
    sys.path.append(str(model_training_path))
    
    from Scripts.Utilities.DataHandler import DataHandler
    from lstm_config import LSTMModelConfig  # Using the correct import
else:
    from ...Utilities.DataHandler import DataHandler
    from ...Scripts.ModelTraining.model_training.models.lstm.lstm_config import LSTMModelConfig

# Set up logging
log_dir = Path("C:/TheTradingRobotPlug/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "lstm_model_trainer.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger('LSTMModelTrainer')

def print_dataset_statistics(data, name):
    logger.info(f"{name} - Shape: {data.shape}")
    logger.info(f"{name} - First few rows:\n{data[:5]}")

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

class LSTMModelTrainer:
    def __init__(self, logger, model_save_path, scaler_save_path):
        self.logger = logger
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path

    def create_sequences(self, X, y, time_steps):
        self.logger.info(f"Creating sequences with time_steps: {time_steps}")
        self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        Xs, ys = [], []
        try:
            for i in range(len(X) - time_steps):
                self.logger.debug(f"Creating sequence for index range: {i} to {i + time_steps}")
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)
        except IndexError as e:
            self.logger.error(f"IndexError in create_sequences: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in create_sequences: {e}")
            raise

    def train_lstm(self, X_train, y_train, X_val, y_val, model_config, epochs, callbacks):
        model = model_config
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
        model.save(self.model_save_path)
        return model

    def evaluate_model(self, X_val, y_val):
        model = tf.keras.models.load_model(self.model_save_path)
        loss = model.evaluate(X_val, y_val)
        self.logger.info(f"Validation loss: {loss}")

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

            trainer = LSTMModelTrainer(logger, model_save_path, scaler_save_path)
            try:
                X_train_seq, y_train_seq = trainer.create_sequences(np.array(X_train), np.array(y_train), time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(np.array(X_val), np.array(y_val), time_steps)

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
                logger.error(f"Error in creating sequences: {e}")
            except KeyError as e:
                logger.error(f"KeyError encountered: {e}")
        else:
            logger.error("Data preprocessing failed.")
    else:
        logger.error("Data loading failed.")

if __name__ == "__main__":
    main()
