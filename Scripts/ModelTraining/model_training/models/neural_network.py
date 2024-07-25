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
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                logger.info("Loaded pretrained model and froze initial layers.")
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
                logger.info("Initialized new model.")

            optimizer_config = self.model_config.get('optimizer', {})
            optimizer_config.pop('type', None)  # Remove 'type' if it exists
            optimizer = Adam(**optimizer_config)
            self.model.compile(optimizer=optimizer, loss=self.model_config.get('loss', 'mse'))
            logger.info("Compiled model with optimizer and loss.")

    def train(self, X_train, y_train, X_val, y_val):
        try:
            self.build_model(X_train.shape[1:])

            early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_config.get('patience', 20), restore_best_weights=True)
            model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
            tensorboard = TensorBoard(log_dir="logs")
            lr_scheduler = LearningRateScheduler(self.scheduler)

            callbacks = [early_stopping, model_checkpoint, tensorboard, lr_scheduler]
            logger.info("Initialized callbacks.")

            # Normalize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.model_config.get('batch_size', 64))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.model_config.get('batch_size', 64))

            self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs, callbacks=callbacks)
            logger.info("Model training completed.")

            y_pred_val = self.model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            explainer = shap.KernelExplainer(self.model.predict, X_train[:100])
            shap_values = explainer.shap_values(X_val[:10])
            shap.summary_plot(shap_values, X_val[:10])

            return self.model
        except Exception as e:
            logger.error("Error during model training.", exc_info=True)
            raise e

class ModelConfig:
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
            'optimizer': {'learning_rate': 0.001},  # Removed 'type': 'adam'
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
            'optimizer': {'learning_rate': 0.001},  # Removed 'type': 'adam'
            'loss': 'mse',
            'batch_size': 64,
            'patience': 20
        }
