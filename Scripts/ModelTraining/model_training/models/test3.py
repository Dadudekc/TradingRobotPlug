import numpy as np
import logging
from multiprocessing import Pool
from time import time
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import pandas as pd
from pathlib import Path

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

from model_training_utils import (
    setup_logger, load_model_from_file, save_predictions, save_metadata, 
    validate_predictions, preprocess_data, detect_models, LoggerHandler
)

try:
    from config_handling import ConfigManager
    from data_store import DataStore
    from Scripts.Utilities.DataHandler import DataHandler
    from Scripts.ModelTraining.model_training.models.basic_lstm_trainer import (
        basicLSTMModelConfig, basicLSTMModelTrainer, prepare_data
    )
    from advanced_lstm_trainer import AdvancedLSTMModelTrainer
    from arima_model_trainer import ARIMAModelTrainer
    from linear_regression_trainer import LinearRegressionModel
    from neural_network_trainer import NeuralNetworkTrainer, NNModelConfig
    from random_forest_trainer import RandomForestModel

except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Helper function to create tooltips
def create_tooltip(widget, text):
    tooltip = tk.Label(widget, text=text, background="yellow", wraplength=200)
    def on_enter(event):
        tooltip.place(x=event.x + 20, y=event.y)
    def on_leave(event):
        tooltip.place_forget()
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

# Class for the main application
class Application(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.title("Automated Model Trainer")
        self.geometry("800x600")
        self.config = config

        self.create_widgets()

    def create_widgets(self):
        # Logging text widget
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.log_text = tk.Text(log_frame, state='disabled', height=10, width=80)
        self.log_text.pack(fill="both", expand=True)
        
        # Schedule dropdown
        schedule_label = ttk.Label(self, text="Schedule:")
        schedule_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')

        self.schedule_dropdown = ttk.Combobox(self, values=["Daily", "Weekly", "Monthly"])
        self.schedule_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        self.schedule_dropdown.current(0)
        create_tooltip(self.schedule_dropdown, "Select the schedule for automated training")

        # Start button
        start_button = ttk.Button(self, text="Start Automated Training", command=self.start_automated_training)
        start_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def log_message(self, message, level="INFO"):
        log_colors = {"INFO": "black", "WARNING": "orange", "ERROR": "red", "DEBUG": "blue"}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp} - {level}] {message}\n"
        self.log_text.config(state='normal')
        self.log_text.tag_config(level, foreground=log_colors.get(level, "black"))
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_automated_training(self):
        try:
            # Example of how to start the automated training
            self.log_message("Starting automated training...")
            trainer = AutomatedModelTrainer(self.config, self.schedule_dropdown, self.log_text, data_handler, model_trainer, model_evaluator, hyperparameter_tuner)
            trainer.start_automated_training()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

# Example configuration
config = {
    "Data": {"file_path": "path/to/data.csv", "y_test": np.array([1, 2, 3]), "y_pred": np.array([1.1, 1.9, 3.2])},
    "Model": {"model_type": "neural_network", "epochs": "10", "param_distributions": {"units": [32, 64, 128], "dropout": [0.1, 0.2, 0.3]}}
}

# Initialize components with DataHandler integration
data_store = DataStore()
data_handler = DataHandler(logger=logging.getLogger("DataHandlerLogger"), data_store=data_store)
model_trainer = ModelTrainer(config, None)
model_evaluator = ModelEvaluator(None)
hyperparameter_tuner = HyperparameterTuner(logger=ModelTrainingLogger(None))

# Start the application
app = Application(config)
app.mainloop()

# Model training and prediction functions (merged)
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
    print("Data path selected: ", data_path)  # Debug print

    try:
        data = data_handler.load_data(data_path)
        logger.info("Data loaded successfully.")
        print("Columns in the DataFrame:", data.columns)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Preprocess data using DataHandler
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(data, target_column='close')

    if X_train is None or y_train is None:
        logger.error("Data preprocessing failed.")
        return

    print(f"Selected features for model training: {X_train.columns.tolist()}")

    # Continue with other models (if any)
    for model_type, model_path in detected_models.items():
        logger.info(f"Processing model type: {model_type}")
        model = load_model_from_file(model_type, model_path, logger)
        
        if model is not None:
            print(f"Features shape: {X_train.shape}")

            preds = model.predict(X_train)
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
    
    # Load and preprocess data using DataHandler
    data = data_handler.load_data(data_file_path)
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(data, target_column='close')

    if X_train is not None and X_val is not None and y_train is not None and y_val is not None:
        trainer = AdvancedLSTMModelTrainer(logger_handler, model_save_path, scaler_save_path)
        trainer.train_lstm(X_train, y_train, X_val, y_val, epochs=50)
    else:
        logger_handler.log("Data preprocessing failed.", "ERROR")

# Linear Regression Model Training function
def train_linear_regression(data_file_path):
    logger_handler = LoggerHandler(logger=logging.getLogger('LinearRegressionModel'))
    
    # Load and preprocess data using DataHandler
    data = data_handler.load_data(data_file_path)
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
    )
    
    if X_train is None or y_train is None:
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
    
    # Load and preprocess data using DataHandler
    data = data_handler.load_data(data_file_path)
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
    )
    
    if X_train is None or y_train is None:
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
    
    # Load and preprocess data using DataHandler
    data = data_handler.load_data(data_file_path)
    X_train, X_val, y_train, y_val = data_handler.preprocess_data(
        data, 
        target_column='close',  # Assuming 'close' is the target column
    )
    
    if X_train is None or y_train is None:
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

# Detect the most recent data file function (now part of the main module)
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
