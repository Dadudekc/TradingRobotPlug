import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class RefinedLSTMModelConfig:
    @staticmethod
    def lstm_model(input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

class RefinedLSTMModelTrainer:
    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def create_sequences(data, target, time_steps=10):
        sequences = []
        targets = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
            targets.append(target[i + time_steps])
        return np.array(sequences), np.array(targets)

    def train_lstm(self, X_train, y_train, X_val, y_val, model_config, epochs=50):
        self.logger.info("Initializing the LSTM model...")
        model = model_config
        
        self.logger.info("Starting model training...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
        self.logger.info("LSTM model training complete")
        
        return model, history

def prepare_data(data, target_column='close', time_steps=10):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    data = data.copy()
    numeric_data = data.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    target = data[target_column].values
    sequences, targets = RefinedLSTMModelTrainer.create_sequences(scaled_data, target, time_steps)
    
    return sequences, targets, scaler

def plot_predictions(y_true_scaled, y_pred_scaled, scaler, target_column):
    # Inverse transform to get back to original scale
    y_true = scaler.inverse_transform(np.concatenate([np.zeros((len(y_true_scaled), scaler.n_features_in_ - 1)), y_true_scaled.reshape(-1, 1)], axis=1))[:, -1]
    y_pred = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1)), y_pred_scaled.reshape(-1, 1)], axis=1))[:, -1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.title('LSTM Model Predictions vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    logger = logging.getLogger("LSTM_Training")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Example data preparation and training
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2020', periods=100),
        'symbol': ['AAPL'] * 100,
        'close': np.random.rand(100) * 100,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'volume': np.random.randint(1000, 10000, size=100)
    })
    target_column = 'close'
    
    X_train_seq, y_train_seq, scaler = prepare_data(data, target_column, time_steps=10)
    X_val_seq, y_val_seq, _ = prepare_data(data, target_column, time_steps=10)
    
    # Ensure data shape is correct
    print("X_train_seq shape:", X_train_seq.shape)
    print("y_train_seq shape:", y_train_seq.shape)
    print("X_val_seq shape:", X_val_seq.shape)
    print("y_val_seq shape:", y_val_seq.shape)
    
    lstm_trainer = RefinedLSTMModelTrainer(logger)
    
    # Make sure the shape of X_train_seq is as expected
    try:
        model_config = RefinedLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    except (IndexError, ValueError) as e:
        logger.error(f"Error in model configuration: {e}")
        raise
    
    model, history = lstm_trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50)
    
    # Make predictions
    predictions = model.predict(X_val_seq)
    
    # Plot predictions vs actual values (unscaled)
    plot_predictions(y_val_seq, predictions, scaler, target_column)
