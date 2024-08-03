import logging
import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move three levels up to the project root (adjust based on your directory structure)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir, os.pardir))

    # Add the project root to the Python path
    sys.path.append(project_root)

    print("Corrected Project root path:", project_root)

from advanced_lstm_trainer import AdvancedLSTMModelTrainer
from basiclstm import basicLSTMModelTrainer, basicLSTMModelConfig, prepare_data
from Scripts.Utilities.model_training_utils import LoggerHandler, DataLoader, DataPreprocessor
from Scripts.Utilities.config_handling import ConfigManager  # Import ConfigManager

def main():
    # Initialize logger
    logger_handler = LoggerHandler()

    # Load and preprocess data
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()  # Assuming ConfigManager is properly defined elsewhere
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Example path to your dataset
    file_path = r"C:\TheTradingRobotPlug\data\alpha_vantage\tsla_data.csv"
    data = data_loader.load_data(file_path)

    if data is None:
        logger_handler.log("Data loading failed. Exiting.", "ERROR")
        return

    target_column = 'close'
    time_steps = 10

    # Preprocess data
    X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(data, target_column=target_column)

    if X_train is None or X_val is None:
        logger_handler.log("Data preprocessing failed. Exiting.", "ERROR")
        return

    # Choose between advanced and basic model
    use_advanced = True  # Set this to False to use the basic model

    if use_advanced:
        logger_handler.log("Using Advanced LSTM Model Trainer")
        trainer = AdvancedLSTMModelTrainer(logger_handler)
        
        # Data preparation for advanced model
        X_train_seq, y_train_seq = trainer.create_sequences(X_train.values, y_train.values, time_steps)
        X_val_seq, y_val_seq = trainer.create_sequences(X_val.values, y_val.values, time_steps)
        
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

        model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

        # Define the callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        # Train the advanced LSTM model with callbacks
        trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, callbacks=[early_stopping, reduce_lr])

    else:
        logger_handler.log("Using Basic LSTM Model Trainer")
        trainer = basicLSTMModelTrainer(logger_handler)
        
        X_train_seq, y_train_seq, scaler = prepare_data(X_train, target_column, time_steps)
        X_val_seq, y_val_seq, _ = prepare_data(X_val, target_column, time_steps)  # Use the same function for validation
        
        model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50)

if __name__ == "__main__":
    main()
