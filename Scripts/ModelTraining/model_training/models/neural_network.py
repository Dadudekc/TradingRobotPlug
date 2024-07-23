import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.metrics import mean_squared_error, r2_score
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNetworkTrainer:
    def __init__(self, model_config, epochs=100, pretrained_model_path=None):
        """
        Initialize the NeuralNetworkTrainer class.

        Parameters:
        - model_config: dict, configuration for the model layers and compilation
        - epochs: int, number of epochs to train (default=100)
        - pretrained_model_path: str, path to a pretrained model (default=None)
        """
        self.model_config = model_config
        self.epochs = epochs
        self.pretrained_model_path = pretrained_model_path
        self.model = None
        self.strategy = tf.distribute.MirroredStrategy()

    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def build_model(self, input_shape):
        """
        Build or load the model based on the configuration.
        
        Parameters:
        - input_shape: tuple, shape of the input data
        """
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
            optimizer = Adam(**optimizer_config)
            self.model.compile(optimizer=optimizer, loss=self.model_config.get('loss', 'mse'))
            logger.info("Compiled model with optimizer and loss.")

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the neural network model.

        Parameters:
        - X_train: np.array, training features
        - y_train: np.array, training labels
        - X_val: np.array, validation features
        - y_val: np.array, validation labels

        Returns:
        - model: trained Keras model
        """
        try:
            self.build_model(X_train.shape[1:])

            early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_config.get('patience', 10), restore_best_weights=True)
            model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
            tensorboard = TensorBoard(log_dir="logs")
            lr_scheduler = LearningRateScheduler(self.scheduler)

            callbacks = [early_stopping, model_checkpoint, tensorboard, lr_scheduler]
            logger.info("Initialized callbacks.")

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.model_config.get('batch_size', 32))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.model_config.get('batch_size', 32))

            self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs, callbacks=callbacks)
            logger.info("Model training completed.")

            y_pred_val = self.model.predict(X_val).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            # Explainability using SHAP
            explainer = shap.DeepExplainer(self.model, X_train[:100])  # Sampled to reduce computation
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
                {'type': 'dense', 'units': 64, 'activation': 'relu'},
                {'type': 'dropout', 'rate': 0.5},
                {'type': 'batch_norm'},
                {'type': 'dense', 'units': 32, 'activation': 'relu'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 32,
            'patience': 10
        }

    @staticmethod
    def lstm_model():
        return {
            'layers': [
                {'type': 'lstm', 'units': 50, 'activation': 'tanh', 'return_sequences': True},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'lstm', 'units': 50, 'activation': 'tanh'},
                {'type': 'dense', 'units': 1, 'activation': 'linear'}
            ],
            'optimizer': {'learning_rate': 0.001},
            'loss': 'mse',
            'batch_size': 32,
            'patience': 10
        }

# Example usage:
# Choose model configuration
model_config = ModelConfig.dense_model()  # or ModelConfig.lstm_model()
trainer = NeuralNetworkTrainer(model_config, epochs=50)
model = trainer.train(X_train, y_train, X_val, y_val)
