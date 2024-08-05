import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def detect_models(model_dir):
    """Detect available models in the specified directory."""
    model_types = ['arima', 'lstm', 'neural_network', 'random_forest', 'linear_regression']
    detected_models = {}
    
    for model_type in model_types:
        model_files = list(Path(model_dir).rglob(f"*{model_type}*"))
        if model_files:
            detected_models[model_type] = str(model_files[0])  # Take the first found model
    
    return detected_models

def load_model_from_file(model_type, model_path, logger):
    try:
        # Implement model loading logic here, e.g., using joblib, pickle, etc.
        model = your_model_loading_function(model_path)
        logger.info(f"Successfully loaded {model_type} model from {model_path}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_type} model from {model_path}: {str(e)}")
        return None


def preprocess_data(data, model_type):
    if model_type in ['lstm', 'neural_network']:
        data = data.reshape((data.shape[0], data.shape[1], 1))
    return data

def save_predictions(predictions, model_type, output_dir, format='parquet', compress=True):
    predictions_df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = os.path.join(output_dir, model_type, timestamp)
    os.makedirs(subdir, exist_ok=True)

    if format == 'csv':
        output_path = os.path.join(subdir, f"{model_type}_predictions.csv")
        predictions_df.to_csv(output_path, index=False)
    elif format == 'json':
        output_path = os.path.join(subdir, f"{model_type}_predictions.json")
        predictions_df.to_json(output_path, orient='records')
    elif format == 'parquet':
        output_path = os.path.join(subdir, f"{model_type}_predictions.parquet")
        predictions_df.to_parquet(output_path, index=False, compression='gzip' if compress else None)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path

def save_metadata(output_dir, model_type, model_path, input_data_path, prediction_path, logger=None):
    metadata = {
        "model_type": model_type,
        "model_path": model_path,
        "input_data_path": input_data_path,
        "prediction_path": prediction_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_df = pd.DataFrame([metadata])
    summary_file = os.path.join(output_dir, "output_summary.csv")

    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        metadata_df = pd.concat([existing_df, metadata_df])

    metadata_df.to_csv(summary_file, index=False)
    if logger:
        logger.info(f"Metadata saved to {summary_file}")

def validate_predictions(predictions, logger=None):
    if any(pred is None or np.isnan(pred).any() for pred in predictions.values()):
        if logger:
            logger.warning("Some predictions contain NaN or None values.")
    else:
        if logger:
            logger.info("All predictions are valid.")
def create_sequences(data, target, time_steps=10):
    sequences = []
    targets = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        targets.append(target[i + time_steps])
    return np.array(sequences), np.array(targets)

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
    sequences, targets = create_sequences(scaled_data, target, time_steps)
    
    return sequences, targets, scaler

def load_config(config_file):
    """Function to load a YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_project_root():
    """Return the project root path based on current file location."""
    return Path(__file__).resolve().parent.parent.parent

