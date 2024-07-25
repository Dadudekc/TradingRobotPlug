import numpy as np
import pandas as pd
from keras.models import load_model, Model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Attention
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import logging
import joblib
import optuna

class LSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape):
        return {
            'input_shape': input_shape,
            'layers': [
                {'type': 'bidirectional_lstm', 'units': 100, 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
                {'type': 'attention'},
                {'type': 'batch_norm'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'dense', 'units': 20, 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)}
            ],
            'optimizer': 'adam',
            'loss': 'mean_squared_error'
        }

class LSTMModelTrainer:
    def __init__(self, logger, model_save_path='best_model.keras', scaler_save_path='scaler.pkl'):
        self.logger = logger
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path

    def preprocess_data(self, X_train, X_val):
        """Preprocess data by handling missing values and scaling."""
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        joblib.dump(scaler, self.scaler_save_path)  # Save the scaler
        return X_train_scaled, X_val_scaled

    def build_model(self, model_config):
        """Build an LSTM model based on the provided configuration."""
        inputs = Input(shape=model_config['input_shape'])
        x = inputs
        
        for layer in model_config['layers']:
            if layer['type'] == 'lstm':
                x = LSTM(units=layer['units'], return_sequences=layer.get('return_sequences', False), kernel_regularizer=layer['kernel_regularizer'])(x)
            elif layer['type'] == 'bidirectional_lstm':
                x = Bidirectional(LSTM(units=layer['units'], return_sequences=layer.get('return_sequences', False), kernel_regularizer=layer['kernel_regularizer']))(x)
            elif layer['type'] == 'batch_norm':
                x = BatchNormalization()(x)
            elif layer['type'] == 'dropout':
                x = Dropout(rate=layer['rate'])(x)
            elif layer['type'] == 'attention':
                x = Attention()([x, x])
            elif layer['type'] == 'dense':
                x = Dense(units=layer['units'], activation=layer['activation'], kernel_regularizer=layer['kernel_regularizer'])(x)

        outputs = Dense(1)(x)  # Output layer to match the target shape
        model = Model(inputs, outputs)
        model.compile(optimizer=model_config['optimizer'], loss=model_config['loss'])
        
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val, model_config, epochs=100, pretrained_model_path=None):
        """Train an LSTM model."""
        self.logger.info("Starting LSTM model training...")
        try:
            model = self.build_model(model_config)
            
            # Log shapes of the input data
            self.logger.info(f"X_train shape before training: {X_train.shape}")
            self.logger.info(f"X_val shape before training: {X_val.shape}")

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True)
            lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32,
                      callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler])

            y_pred_val = model.predict(X_val).flatten()
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
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            # Make predictions
            y_pred_test = model.predict(X_test_scaled).flatten()

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)

            self.logger.info(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")

def create_sequences(data, target, time_steps=1):
    xs, ys = [], []
    for i in range(len(data) - time_steps):
        x = data[i:(i + time_steps)]
        y = target[i + time_steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def objective(trial):
    model_config = {
        'input_shape': (time_steps, len(selected_features)),  # Adjust input shape to include time steps and number of features
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
    model = trainer.train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=50)
    y_pred_val = model.predict(X_val).flatten()
    mse = mean_squared_error(y_val, y_pred_val)
    return mse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load your data here
    # Assuming the data has been preprocessed to include the selected features
    data = pd.read_csv('/mnt/data/tsla_data.csv')
    # Selecting a subset of features for this example
    selected_features = ['open', 'high', 'low', 'close', 'volume']
    data = data[selected_features + ['date']]
    data = data.dropna()  # Dropping rows with any NaN values

    # Splitting data into training, validation, and test sets
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Convert data to numpy arrays
    X_train = train_data[selected_features].values
    y_train = train_data['close'].values
    X_val = val_data[selected_features].values
    y_val = val_val = val_data['close'].values
    X_test = test_data[selected_features].values
    y_test = test_data['close'].values

    # Create sequences for LSTM model
    time_steps = 10

    X_train, y_train = create_sequences(X_train, y_train, time_steps)
    X_val, y_val = create_sequences(X_val, y_val, time_steps)
    X_test, y_test = create_sequences(X_test, y_test, time_steps)

    # Initialize the trainer
    trainer = LSTMModelTrainer(logger)

    # Preprocess data
    X_train_scaled, X_val_scaled = trainer.preprocess_data(X_train, X_val)

    # Hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Train the final model with the best hyperparameters
    best_model_config = {
        'input_shape': (time_steps, len(selected_features)),  # Adjust input shape to include time steps and number of features
        'layers': [
            {'type': 'bidirectional_lstm', 'units': study.best_params['units_lstm'], 'return_sequences': True, 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)},
            {'type': 'attention'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': study.best_params['dropout_rate']},
            {'type': 'dense', 'units': study.best_params['units_dense'], 'activation': 'relu', 'kernel_regularizer': l1_l2(l1=0.01, l2=0.01)}
        ],
        'optimizer': study.best_params['optimizer'],
        'loss': 'mean_squared_error'
    }

    # Re-preprocess data with the full dataset
    X_train_scaled, X_val_scaled = trainer.preprocess_data(X_train, X_val)

    # Train and evaluate the final model
    final_model = trainer.train_lstm(X_train_scaled, y_train, X_val_scaled, y_val, best_model_config, epochs=50)
    trainer.evaluate_model(X_test, y_test)

