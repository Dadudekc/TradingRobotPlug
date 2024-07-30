import logging
import sys
from pathlib import Path
import pandas as pd

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.DataHandler import DataHandler
from Scripts.ModelTraining.model_training.models.lstm_trainer import LSTMModelTrainer, LSTMModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LSTMModelTrainer')

# File paths
data_file_path = 'C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv'
model_save_path = 'best_model.keras'
scaler_save_path = 'scaler.pkl'

# Initialize DataHandler
data_handler = DataHandler(logger=logger)

# Load and preprocess data
data = data_handler.load_data(data_file_path)
if data is not None:
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(data)

    if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
        # Create sequences
        time_steps = 10
        trainer = LSTMModelTrainer(logger, model_save_path, scaler_save_path)
        X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
        X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

        # Debug: Print shapes
        logger.info(f"X_train_seq shape: {X_train_seq.shape}")
        logger.info(f"y_train_seq shape: {y_train_seq.shape}")
        logger.info(f"X_val_seq shape: {X_val_seq.shape}")
        logger.info(f"y_val_seq shape: {y_val_seq.shape}")

        # Define model parameters
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

        # Initialize and train the model
        model_config = LSTMModelConfig.lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), model_params)
        trained_model = trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50)

        # Evaluate the model if training was successful
        if trained_model:
            trainer.evaluate_model(X_val_seq, y_val_seq)
    else:
        logger.error("Data preprocessing failed.")
else:
    logger.error("Data loading failed.")
