import os
import sys
import pandas as pd
from multiprocessing import Pool
from time import time
from pathlib import Path
import numpy as np

# Adjust import path based on your project structure
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Assuming project root is two levels up

# Add the correct directories to sys.path
model_training_dir = project_root / 'ModelTraining' / 'model_training'
utilities_dir = project_root / 'Utilities'

sys.path.append(str(model_training_dir))
sys.path.append(str(utilities_dir))

# Debug print to confirm the paths
print("Corrected Project root path:", project_root)
print("Adding ModelTraining directory to sys.path:", model_training_dir)
print("Adding Utilities directory to sys.path:", utilities_dir)

# Now import the DataStore class
try:
    from data_store import DataStore
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Import the basic LSTM model and trainer
try:
    from Scripts.ModelTraining.model_training.models.basic_lstm_trainer import basicLSTMModelConfig, basicLSTMModelTrainer, prepare_data
except ModuleNotFoundError as e:
    print(f"Error importing LSTM model modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Other imports
from test2 import setup_logger, load_model_from_file, save_predictions, save_metadata, validate_predictions, preprocess_data, detect_models
from arima_model_trainer import ARIMAModelTrainer

from linear_regression_trainer import LinearRegressionModel
from neural_network_trainer import NeuralNetworkTrainer, NNModelConfig
from random_forest_trainer import RandomForestModel

# ARIMA training function
def train_arima(symbol, threshold=100):
    arima_trainer = ARIMAModelTrainer(symbol=symbol, threshold=threshold)
    arima_trainer.train()

# Prediction generation function
def generate_predictions(model_dir, data_dir, output_format='parquet', output_dir='output', parallel=False):
    logger = setup_logger("Prediction_Generator")

    # Automatically detect available models
    detected_models = detect_models(model_dir)
    if not detected_models:
        logger.error("No models detected in the specified directory.")
        return

    logger.info(f"Detected models: {detected_models}")

    # Automatically detect the data file
    data_path = detect_data_file(data_dir)
    if data_path is None or not os.path.exists(data_path):
        logger.error(f"No valid data file found in directory: {data_dir}")
        return

    logger.info(f"Using data file: {data_path}")
    data = pd.read_csv(data_path)
    logger.info("Data loaded successfully.")
    print("Columns in the DataFrame:", data.columns)

    # Exclude non-numeric or unwanted columns (e.g., 'date', 'symbol')
    excluded_columns = ['date', 'symbol']
    features = data.drop(columns=excluded_columns).select_dtypes(include=[float, int])

    print(f"Selected features for model training: {features.columns.tolist()}")

    # Check for NaN or Inf values in the features
    try:
        check_for_nan_inf(features.values)
    except ValueError as e:
        logger.error(e)
        return

    # Prepare data for LSTM model
    X_train, y_train, scaler = prepare_data(data, target_column='close', time_steps=10)
    X_val, y_val, _ = prepare_data(data, target_column='close', time_steps=10)
    
    # Initialize and train the LSTM model
    lstm_trainer = basicLSTMModelTrainer(logger)
    model_config = basicLSTMModelConfig.lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model, history = lstm_trainer.train_lstm(X_train, y_train, X_val, y_val, model_config, epochs=50)
    
    logger.info("LSTM model training completed.")
    # Predictions using the LSTM model
    lstm_predictions = model.predict(X_val)
    predictions = {"LSTM": lstm_predictions}

    # Save LSTM model predictions
    prediction_path = save_predictions(lstm_predictions, "LSTM", output_dir, format=output_format)
    save_metadata(output_dir, "LSTM", detected_models.get("LSTM", "N/A"), data_path, prediction_path, logger)

    logger.info(f"Predictions saved at {prediction_path}")

    # Continue with other models (if any)
    if parallel:
        with Pool() as pool:
            prediction_results = pool.map(parallel_generate_predictions, [
                (model_type, model_path, features.values, logger)
                for model_type, model_path in detected_models.items()
            ])
            predictions.update(dict(prediction_results))
    else:
        for model_type, model_path in detected_models.items():
            logger.info(f"Processing model type: {model_type}")
            model = load_model_from_file(model_type, model_path, logger)
            
            if model is not None:
                preds = model.predict(preprocess_data(features.values, model_type))
                predictions[model_type] = preds
                logger.info(f"Predictions for {model_type}: {preds[:5]}")
            else:
                logger.error(f"Skipping predictions for {model_type} due to loading error.")

    validate_predictions(predictions, logger)

    for model_type, preds in predictions.items():
        prediction_path = save_predictions(preds, model_type, output_dir, format=output_format)
        save_metadata(output_dir, model_type, detected_models[model_type], data_path, prediction_path, logger)

# Advanced LSTM Model Training function
def train_advanced_lstm(data_file_path, model_save_path="best_model.keras", scaler_save_path="scaler.pkl"):
    logger_handler = LoggerHandler(logger=logging.getLogger('AdvancedLSTMModelTrainer'))
    data_loader = DataLoader(logger_handler)
    config_manager = None  # Replace with actual ConfigManager if available
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load data using DataLoader
    data = data_loader.load_data(data_file_path)

    if data is not None:
        X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(data)

        if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
            time_steps = 10

            logger_handler.log(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
            logger_handler.log(f"X_val length: {len(X_val)}, y_val length: {len(y_val)}")

            trainer = AdvancedLSTMModelTrainer(logger_handler, model_save_path, scaler_save_path)
            try:
                logger_handler.log(f"Creating sequences with time_steps: {time_steps}")
                X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, time_steps)

                logger_handler.log(f"X_train_seq shape: {X_train_seq.shape}")
                logger_handler.log(f"y_train_seq shape: {y_train_seq.shape}")
                logger_handler.log(f"X_val_seq shape: {X_val_seq.shape}")
                logger_handler.log(f"y_val_seq shape: {y_val_seq.shape}")

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

                trained_model = trainer.train_lstm(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50,
                    callbacks=[early_stopping, reduce_lr]
                )

                if trained_model:
                    trainer.evaluate_model(X_val_seq, y_val_seq)
            except ValueError as e:
                logger_handler.log(f"ValueError in creating sequences: {e}", "ERROR")
            except KeyError as e:
                logger_handler.log(f"KeyError encountered: {e}", "ERROR")
        else:
            logger_handler.log("Data preprocessing failed.", "ERROR")
    else:
        logger_handler.log("Data loading failed.", "ERROR")

# Linear Regression Model Training function
def train_linear_regression(data_file_path):
    logger_handler = LoggerHandler(logger=logging.getLogger('LinearRegressionModel'))
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load the data
    data = data_loader.load_data(data_file_path)

    if data is None:
        logger_handler.log("Failed to load data. Exiting.", "ERROR")
        return
    
    # Preprocess the data
    X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
        date_column='date',      # Assuming 'date' is the date column
        lag_sizes=[1, 2, 3, 5, 10], 
        window_sizes=[5, 10, 20],
        scaler_type='StandardScaler'
    )
    
    if X_train is None or X_val is None:
        logger_handler.log("Data preprocessing failed. Exiting.", "ERROR")
        return
    
    # Instantiate the model
    model = LinearRegressionModel(logger=logger_handler.logger)
    
    # Train the model with explainability
    best_model = model.train_with_explainability(X_train, y_train, X_val, y_val)
    
    if best_model:
        logger_handler.log("Model training and explainability completed successfully.")
    else:
        logger_handler.log("Model training failed.", "ERROR")

# Neural Network Model Training function
def train_neural_network(data_file_path, model_config_name="dense_model"):
    logger_handler = LoggerHandler(logger=logging.getLogger('NeuralNetworkModel'))
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load the data
    data = data_loader.load_data(data_file_path)

    if data is None:
        logger_handler.log("Failed to load data. Exiting.", "ERROR")
        return
    
    # Preprocess the data
    X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
        date_column='date',      # Assuming 'date' is the date column
        lag_sizes=[1, 2, 3, 5, 10], 
        window_sizes=[5, 10, 20],
        scaler_type='StandardScaler'
    )
    
    if X_train is None or X_val is None:
        logger_handler.log("Data preprocessing failed. Exiting.", "ERROR")
        return
    
    # Choose the model configuration
    if model_config_name == "dense_model":
        model_config = NNModelConfig.dense_model()
    elif model_config_name == "lstm_model":
        model_config = NNModelConfig.lstm_model()
    else:
        logger_handler.log(f"Unknown model configuration: {model_config_name}. Exiting.", "ERROR")
        return
    
    # Instantiate the trainer
    trainer = NeuralNetworkTrainer(model_config=model_config, epochs=50)
    
    # Train the model
    trained_model = trainer.train(X_train, y_train, X_val, y_val)
    
    if trained_model:
        logger_handler.log("Neural network training completed successfully.")
    else:
        logger_handler.log("Neural network training failed.", "ERROR")

# Random Forest Model Training function
def train_random_forest(data_file_path):
    logger_handler = LoggerHandler(logger=logging.getLogger('RandomForestModel'))
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)

    # Load the data
    data = data_loader.load_data(data_file_path)

    if data is None:
        logger_handler.log("Failed to load data. Exiting.", "ERROR")
        return
    
    # Preprocess the data
    X_train, X_val, y_train, y_val = data_preprocessor.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
        date_column='date',      # Assuming 'date' is the date column
        lag_sizes=[1, 2, 3, 5, 10], 
        window_sizes=[5, 10, 20],
        scaler_type='StandardScaler'
    )
    
    if X_train is None or X_val is None:
        logger_handler.log("Data preprocessing failed. Exiting.", "ERROR")
        return
    
    # Instantiate the model
    model = RandomForestModel(logger=logger_handler.logger)
    
    # Train the model
    best_model, best_params, mse, rmse, mae, mape, r2 = model.train(X_train, y_train)
    
    if best_model:
        logger_handler.log("Random Forest model training completed successfully.")
        logger_handler.log(f"Best Parameters: {best_params}")
        logger_handler.log(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R²: {r2}")
    else:
        logger_handler.log("Random Forest model training failed.", "ERROR")

# Detect the most recent data file
def detect_data_file(data_dir, file_extension='csv'):
    data_files = list(Path(data_dir).rglob(f"*.{file_extension}"))
    if not data_files:
        return None
    
    data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(data_files) > 1:
        print("Multiple data files found. Please choose one:")
        for i, file in enumerate(data_files, 1):
            print(f"{i}: {file.name}")
        choice = int(input("Enter the number of the file to use: ")) - 1
        return str(data_files[choice])
    
    return str(data_files[0])

# Main execution
if __name__ == "__main__":
    start_time = time()
    
    # Example usage: Train ARIMA model
    train_arima(symbol="AAPL", threshold=10)
    
    # Example usage: Generate predictions
    generate_predictions(model_dir='models', data_dir='data', output_format='parquet', output_dir='output', parallel=False)
    
    # Example usage: Train Advanced LSTM Model
    train_advanced_lstm(data_file_path='C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv')
    
    # Example usage: Train Linear Regression Model
    train_linear_regression(data_file_path='C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv')

    # Example usage: Train Neural Network Model
    train_neural_network(data_file_path='C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv', model_config_name="dense_model")
    
    # Example usage: Train Random Forest Model
    train_random_forest(data_file_path='C:/TheTradingRobotPlug/data/alpha_vantage/tsla_data.csv')
    
    end_time = time()
    print(f"Execution time: {end_time - start_time} seconds")
