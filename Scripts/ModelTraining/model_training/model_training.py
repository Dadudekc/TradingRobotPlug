import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import tsfresh
import logging
import tensorflow as tf
import joblib

class LSTMModelTrainer:
    def __init__(self, logger, model_save_path='best_model.keras', scaler_save_path='scaler.pkl'):
        self.logger = logger
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path

    def preprocess_data(self, X_train, X_val):
        """Preprocess data by handling missing values and scaling."""
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        joblib.dump(scaler, self.scaler_save_path)  # Save the scaler
        return X_train_scaled, X_val_scaled

    def extract_features(self, X_train, X_val):
        """Automated feature extraction using tsfresh."""
        def prepare_tsfresh_data(X):
            """Prepare data for tsfresh."""
            df = pd.DataFrame(X)
            df['id'] = np.arange(len(df))
            df = df.melt(id_vars=['id'], var_name='time', value_name='value')
            return df

        X_train_tsfresh = prepare_tsfresh_data(X_train)
        X_val_tsfresh = prepare_tsfresh_data(X_val)

        X_train_features = tsfresh.extract_features(X_train_tsfresh, column_id='id', column_sort='time')
        X_val_features = tsfresh.extract_features(X_val_tsfresh, column_id='id', column_sort='time')
        
        return X_train_features, X_val_features

    def augment_data(self, X_train, y_train):
        """Time series data augmentation."""
        augmented_data = X_train.copy()
        # Implement your data augmentation techniques here
        return augmented_data, y_train

    def train_lstm(self, X_train, y_train, X_val, y_val, model_config, epochs=100, pretrained_model_path=None):
        """Train an LSTM model."""
        self.logger.info("Starting LSTM model training...")
        try:
            if pretrained_model_path:
                model = load_model(pretrained_model_path)
                for layer in model.layers[:-5]:
                    layer.trainable = False
            else:
                model = Sequential()

            model.add(Input(shape=(X_train.shape[1], 1)))
            for layer in model_config['layers']:
                if layer['type'] == 'lstm':
                    model.add(LSTM(units=layer['units'], return_sequences=layer['return_sequences'], kernel_regularizer=layer['kernel_regularizer']))
                elif layer['type'] == 'batch_norm':
                    model.add(BatchNormalization())
                elif layer['type'] == 'dropout':
                    model.add(Dropout(rate=layer['rate']))
                elif layer['type'] == 'dense':
                    model.add(Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer']))

            model.add(Dense(1))  # Output layer to match the target shape

            model.compile(optimizer=model_config['optimizer'], loss=model_config['loss'])

            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True)

            model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping, reduce_lr, checkpoint])

            y_pred_val = model.predict(X_val_reshaped).flatten()
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            self.logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

            return model

        except Exception as e:
            self.logger.error(f"Error occurred during model training: {e}")
            return None

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance on the test dataset."""
        self.logger.info("Evaluating model on test data...")
        try:
            # Load the model
            model = load_model(self.model_save_path)
            scaler = joblib.load(self.scaler_save_path)

            # Preprocess the test data
            X_test_scaled = scaler.transform(X_test)
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

            # Make predictions
            y_pred_test = model.predict(X_test_reshaped).flatten()

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)

            self.logger.info(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Example data
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)

    trainer = LSTMModelTrainer(logger)

    # Preprocess data
    X_train_scaled, X_val_scaled = trainer.preprocess_data(X_train, X_val)

    # Train model
    model_config = {
        'layers': [
            {'type': 'lstm', 'units': 100, 'return_sequences': False, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 20, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)}
        ],
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    }
    trainer.train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, model_config, epochs=50)

    # Evaluate model
    trainer.evaluate_model(X_test, y_test)
