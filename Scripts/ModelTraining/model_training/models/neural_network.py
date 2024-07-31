import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os

# Configure logging
log_dir = 'C:/TheTradingRobotPlug/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'model_training.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
model_logger = logging.getLogger('neural_network')

class NeuralNetworkTrainer:
    def __init__(self, model_config, epochs=100, pretrained_model_path=None):
        self.model_config = model_config
        self.epochs = epochs
        self.pretrained_model_path = pretrained_model_path
        self.model = None
        self.strategy = tf.distribute.MirroredStrategy()

    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))

    def build_model(self, input_shape):
        with self.strategy.scope():
            if self.pretrained_model_path:
                self.model = load_model(self.pretrained_model_path)
                for layer in self.model.layers[:-5]:
                    layer.trainable = False
                model_logger.info("Loaded pretrained model and froze initial layers.")
            else:
                self.model = Sequential()
                self.model.add(Input(shape=input_shape))
                for layer in self.model_config['layers']:
                    if layer['type'] == 'dense':
                        self.model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer.get('kernel_regularizer', None)))
                    elif layer['type'] == 'batch_norm':
                        self.model.add(BatchNormalization())
                    elif layer['type'] == 'dropout':
                        self.model.add(Dropout(rate=layer['rate']))
                    elif layer['type'] == 'lstm':
                        self.model.add(LSTM(units=layer['units'], activation=layer['activation'], return_sequences=layer.get('return_sequences', False)))
                model_logger.info("Initialized new model.")

            optimizer = Adam(learning_rate=self.model_config.get('optimizer', {}).get('learning_rate', 0.001))
            self.model.compile(optimizer=optimizer, loss=self.model_config.get('loss', 'mse'))
            model_logger.info("Compiled model with optimizer and loss.")

    def train(self, X_train, y_train, X_val, y_val):
        try:
            model_logger.info(f"X_train shape: {X_train.shape}")
            model_logger.info(f"y_train shape: {y_train.shape}")
            model_logger.info(f"X_val shape: {X_val.shape}")
            model_logger.info(f"y_val shape: {y_val.shape}")

            self.build_model(X_train.shape[1:])

            early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_config.get('patience', 20), restore_best_weights=True)
            model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss', mode='min')
            tensorboard = TensorBoard(log_dir="logs")
            lr_scheduler = LearningRateScheduler(self.scheduler)

            callbacks = [early_stopping, model_checkpoint, tensorboard, lr_scheduler]
            model_logger.info("Initialized callbacks.")

            # Normalize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.model_config.get('batch_size', 64))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.model_config.get('batch_size', 64))

            self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs, callbacks=callbacks)
            model_logger.info("Model training completed.")

            y_pred_val = self.model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            model_logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            explainer = shap.KernelExplainer(self.model.predict, X_train[:100])
            shap_values = explainer.shap_values(X_val[:10])
            shap.summary_plot(shap_values, X_val[:10])

            return self.model
        except Exception as e:
            model_logger.error("Error during model training.", exc_info=True)
            raise e

class NNModelConfig:
    @staticmethod
    def dense_model():
        return {
            'layers': [
                {'type': 'dense', 'units': 128, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'batch_norm'},
                {'type': 'dense', 'units': 64, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'dense', 'units': 32, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }

    @staticmethod
    def lstm_model():
        return {
            'layers': [
                {'type': 'lstm', 'units': 100, 'activation': 'tanh', 'return_sequences': True, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'lstm', 'units': 100, 'activation': 'tanh', 'return_sequences': False, 'kernel_regularizer': 'l2'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'dense', 'units': 50, 'activation': 'relu', 'kernel_regularizer': 'l2'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }
# models/neural_network_trainer.py

# (The existing code remains unchanged)

import unittest

class TestNeuralNetworkTrainer(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        self.X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y_train = np.random.rand(100)  # 100 target values
        self.X_val = np.random.rand(20, 10)  # 20 validation samples, 10 features
        self.y_val = np.random.rand(20)  # 20 validation target values

        self.model_config = NNModelConfig.dense_model()  # Using the dense model configuration for the test
        self.trainer = NeuralNetworkTrainer(model_config=self.model_config, epochs=2)

    def test_train(self):
        """Test the training process."""
        try:
            model = self.trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)
            self.assertIsNotNone(model, "The model training did not return a valid model.")
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
