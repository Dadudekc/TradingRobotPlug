import logging
import pandas as pd
from pathlib import Path
import sys

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[0]  # Adjust this according to the correct level
    sys.path.append(str(project_root))

from advanced_lstm_trainer import AdvancedLSTMModelTrainer
from basic_lstm_trainer import basicLSTMModelTrainer, basicLSTMModelConfig, prepare_data

def main():
    # Set up logging
    logger = logging.getLogger("LSTM_Training")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Example data preparation
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
    time_steps = 10

    # Choose between advanced and basic model
    use_advanced = True  # Set this to False to use the basic model

    if use_advanced:
        logger.info("Using Advanced LSTM Model Trainer")
        trainer = AdvancedLSTMModelTrainer(logger)
        
        # Data preparation for advanced model
        # Assuming you have the data split logic defined elsewhere
        X_train, X_val, y_train, y_val = ... # Define your data splitting logic here
        X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
        X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)
        
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
        trainer.train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_config, epochs=50)
    else:
        logger.info("Using Basic LSTM Model Trainer")
        trainer = basicLSTMModelTrainer(logger)
        
        X_train, y_train, scaler = prepare_data(data, target_column, time_steps)
        X_val, y_val, _ = prepare_data(data, target_column, time_steps)  # Replace with actual validation data
        
        model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        trainer.train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=50)

if __name__ == "__main__":
    main()
