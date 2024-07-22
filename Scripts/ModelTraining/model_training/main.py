import pandas as pd
import os
import sys
import logging
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from data_preprocessing import DataPreprocessing
from model_training import ModelTraining

# Setup logger
logger = logging.getLogger('Model_Training')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def main():
    # Initialize Tkinter and hide the main window
    root = Tk()
    root.withdraw()
    
    # Set default directory
    default_dir = os.path.join(project_root, 'data', 'alpha_vantage')
    
    # Ask the user to select a file
    data_file = askopenfilename(initialdir=default_dir, title="Select data file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    logger.info(f"Data file path: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
        logger.info(f"Original data shape: {df.shape}")
        nan_counts = df.isna().sum()
        logger.info(f"NaN values per column:\n{nan_counts}")
    
        # Initialize DataPreprocessing and preprocess data
        data_preprocessor = DataPreprocessing(logger)
        X, y = data_preprocessor.preprocess_data_with_feature_engineering(df)
    
        if X is not None and y is not None:
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
            # Initialize ModelTraining and start training
            model_trainer = ModelTraining(logger)
            model_trainer.start_training(X_train, y_train, X_val, y_val, model_type='ARIMA')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
