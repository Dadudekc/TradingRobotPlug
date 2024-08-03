import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from multiprocessing import Pool
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# Function to load a model based on type
def load_model_from_file(model_type, model_path):
    try:
        if model_type in ['arima', 'random_forest', 'linear_regression']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_type in ['lstm', 'neural_network']:
            model = load_model(model_path)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        logging.info(f"Successfully loaded {model_type} model.")
        return model
    except Exception as e:
        logging.error(f"Error loading {model_type} model from {model_path}: {str(e)}")
        raise

# Function to preprocess data based on model type (if needed)
def preprocess_data(data, model_type):
    if model_type in ['lstm', 'neural_network']:
        data = data.reshape((data.shape[0], data.shape[1], 1))
    return data

# Function to generate predictions
def generate_predictions(model, model_type, data):
    try:
        data = preprocess_data(data, model_type)
        predictions = model.predict(data)
        logging.info(f"Predictions generated using {model_type} model.")
        return predictions
    except Exception as e:
        logging.error(f"Error generating predictions with {model_type} model: {str(e)}")
        raise

# Function to save predictions in efficient formats
def save_predictions(predictions, model_type, output_dir, format='parquet', compress=True):
    try:
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

        logging.info(f"Predictions successfully saved to {output_path} in {format} format.")
        return output_path
    except Exception as e:
        logging.error(f"Error saving predictions: {str(e)}")
        raise

# Function to save metadata
def save_metadata(output_dir, model_type, model_path, input_data_path, prediction_path):
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
    logging.info(f"Metadata saved to {summary_file}")

# Function to validate predictions
def validate_predictions(predictions):
    if any(pred is None or np.isnan(pred).any() for pred in predictions.values()):
        logging.warning("Some predictions contain NaN or None values.")
    else:
        logging.info("All predictions are valid.")

# Parallelized prediction generation
def parallel_generate_predictions(args):
    model_type, model_path, data = args
    model = load_model_from_file(model_type, model_path)
    return model_type, generate_predictions(model, model_type, data)

def main():
    parser = argparse.ArgumentParser(description="Generate predictions using various models.")
    parser.add_argument('--output_format', type=str, default='parquet', help='Format for the output predictions (parquet, csv, json)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory where predictions and metadata will be saved')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing for predictions')

    args = parser.parse_args()

    try:
        # Load models information from environment variables
        models_info = {
            'arima': os.getenv('ARIMA_MODEL_PATH'),
            'lstm': os.getenv('LSTM_MODEL_PATH'),
            'neural_network': os.getenv('NN_MODEL_PATH'),
            'random_forest': os.getenv('RF_MODEL_PATH'),
            'linear_regression': os.getenv('LR_MODEL_PATH')
        }

        # Load prediction data
        data_path = os.getenv('DATA_PATH')
        data = pd.read_csv(data_path)

        # Select features based on environment variable
        features = data[os.getenv('FEATURES').split(',')]

        predictions = {}

        if args.parallel:
            with Pool() as pool:
                prediction_results = pool.map(parallel_generate_predictions, [(model_type, model_path, features.values) for model_type, model_path in models_info.items() if model_path])
                predictions = dict(prediction_results)
        else:
            for model_type, model_path in models_info.items():
                if model_path is None:
                    logging.warning(f"Model path for {model_type} is not set. Skipping.")
                    continue

                logging.info(f"Processing model type: {model_type}")
                model = load_model_from_file(model_type, model_path)
                preds = generate_predictions(model, model_type, features.values)
                predictions[model_type] = preds
                logging.info(f"Predictions for {model_type}: {preds[:5]}")  # Log the first 5 predictions

        validate_predictions(predictions)

        # Save predictions and metadata
        for model_type, preds in predictions.items():
            prediction_path = save_predictions(preds, model_type, args.output_dir, format=args.output_format)
            save_metadata(args.output_dir, model_type, models_info[model_type], data_path, prediction_path)

    except Exception as e:
        logging.error(f"An error occurred in the prediction process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logging.info(f"Execution time: {end_time - start_time} seconds")
