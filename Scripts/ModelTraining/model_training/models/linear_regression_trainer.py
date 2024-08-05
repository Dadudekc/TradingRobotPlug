# File: linear_regression.py
# Location: Scripts/ModelTraining/model_training/models
# Description: Contains a Linear Regression model class with training, explainability, and streaming capabilities.

import os
import sys
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

# Scikit-learn imports
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Dynamic Root Path Setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # Assuming project root is three levels up

# Add the 'Utilities' directory to sys.path
utilities_dir = project_root / 'Scripts' / 'Utilities'
sys.path.append(str(utilities_dir))

# Debug print to confirm the path
print("Corrected Project root path:", project_root)
print("Adding Utilities directory to sys.path:", utilities_dir)

# Now, import the required modules from Utilities
from model_training_utils import LoggerHandler, DataLoader, DataPreprocessor

# Define the directory for LIME explanations
lime_explanation_dir = os.path.join(project_root, 'lime_explanations')

# Create the directory if it doesn't exist
os.makedirs(lime_explanation_dir, exist_ok=True)

# Logging setup
log_path = os.path.join(project_root, 'logs')
os.makedirs(log_path, exist_ok=True)

log_file = os.path.join(log_path, 'application.log')
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now you can import config_handling
try:
    from config_handling import ConfigManager
except ImportError as e:
    logger.error(f"Failed to import ConfigManager from config_handling: {e}")
    raise




class LinearRegressionModel:
    def __init__(self, logger=None):
        self.logger = logger
        self.best_model = None
        self.selector = None

    def train(self, X_train, y_train, X_val, y_val):
        """Train a linear regression model with hyperparameter tuning and feature selection."""
        if self.logger:
            self.logger.info(f"Initial X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            self.logger.info(f"Initial X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        ridge_for_selection = Ridge(alpha=1.0)
        self.selector = SelectFromModel(estimator=ridge_for_selection, threshold='mean')

        # Fit the selector on the training data
        self.selector.fit(X_train, y_train)

        # Transform the training and validation data
        X_train_selected = self.selector.transform(X_train)
        X_val_selected = self.selector.transform(X_val)

        if self.logger:
            self.logger.info(f"X_train shape after feature selection: {X_train_selected.shape}")
            self.logger.info(f"X_val shape after feature selection: {X_val_selected.shape}")

        # Create a pipeline for scaling and regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        param_grid = {'ridge__alpha': np.logspace(-4, 0, 100)}

        if self.logger:
            self.logger.info("Starting randomized search...")

        randomized_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=50,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )

        try:
            randomized_search.fit(X_train_selected, y_train)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during model training: {str(e)}")
            return None

        if self.logger:
            self.logger.info("Randomized Search Results:")
            results_df = pd.DataFrame(randomized_search.cv_results_)
            results_str = results_df[['param_ridge__alpha', 'mean_test_score', 'std_test_score']].to_string()
            self.logger.info(results_str)

        cv_scores = cross_val_score(randomized_search.best_estimator_, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
        if self.logger:
            cv_scores_str = ", ".join([f"{score:.2f}" for score in cv_scores])
            self.logger.info(f"CV Scores: {cv_scores_str}")

        self.best_model = randomized_search.best_estimator_

        if self.logger:
            self.logger.info(f"Best model's training pipeline steps: {self.best_model.named_steps}")

        # Predict and calculate metrics on transformed validation data
        y_pred_val = self.best_model.predict(X_val_selected)
        if self.logger:
            self.logger.info(f"Predicted y_val shape: {y_pred_val.shape}")

        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)  # Calculate RMSE by taking the square root of MSE
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        if self.logger:
            self.logger.info(f"Validation Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            best_alpha = randomized_search.best_params_['ridge__alpha']
            self.logger.info(f"Best regularization strength (alpha): {best_alpha:.4f}. Consider using this as a starting point for your next training session.")

        return self.best_model


    def explain_with_shap(self, X_train, X_val):
        X_train_selected = self.selector.transform(X_train)
        X_val_selected = self.selector.transform(X_val)
        
        explainer = shap.Explainer(self.best_model.named_steps['ridge'], X_train_selected)
        shap_values = explainer(X_val_selected)

        if self.logger:
            self.logger.info("SHAP Summary Plot saved as shap_summary_plot.png")
        shap.summary_plot(shap_values.values, X_val_selected, show=False)
        plt.savefig('shap_summary_plot.png')

        if self.logger:
            self.logger.info("SHAP Dependence Plot saved as shap_dependence_plot.png")
        shap.dependence_plot(0, shap_values.values, X_val_selected, show=False)
        plt.savefig('shap_dependence_plot.png')

    def explain_with_lime(self, X_train, X_val):
        X_train_selected = self.selector.transform(X_train)
        X_val_selected = self.selector.transform(X_val)
        
        explainer = LimeTabularExplainer(X_train_selected, mode='regression')
        for i in range(len(X_val_selected)):
            exp = explainer.explain_instance(X_val_selected[i], self.best_model.predict, num_features=5)
            if self.logger:
                self.logger.info(f"LIME Explanation for instance {i} saved as lime_explanation_{i}.html")
            
            # Save the explanation file in the 'lime_explanations' directory
            explanation_path = os.path.join(lime_explanation_dir, f'lime_explanation_{i}.html')
            exp.save_to_file(explanation_path)


    def train_with_explainability(self, X_train, y_train, X_val, y_val):
        self.train(X_train, y_train, X_val, y_val)
        if self.best_model is not None:
            self.explain_with_shap(X_train, X_val)
            self.explain_with_lime(X_train, X_val)
        return self.best_model

    def consume_streaming_data(self, topic):
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        for message in consumer:
            data = pd.DataFrame([message.value])
            transformed_data = self.selector.transform(data)  # Transform the incoming data using the fitted selector
            prediction = self.best_model.predict(transformed_data)
            if self.logger:
                self.logger.info(f"Real-time Prediction: {prediction}")

    def train_with_automl(self, X_train, y_train, X_val, y_val):
        tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, cv=5)

        try:
            tpot.fit(X_train, y_train)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during TPOT training: {str(e)}")
            return None

        y_pred_val = tpot.predict(X_val)
        if self.logger:
            self.logger.info(f"Predicted y_val shape: {y_pred_val.shape}")

        mse = mean_squared_error(y_val, y_pred_val)
        rmse = root_mean_squared_error(y_val, y_pred_val)  # Updated to use the new function
        r2 = r2_score(y_val, y_pred_val)

        if self.logger:
            self.logger.info(f"Validation Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return tpot.fitted_pipeline_

def main():
    # Initialize logger
    logger_handler = LoggerHandler()
    
    # Initialize data loader and preprocessor
    data_loader = DataLoader(logger_handler)
    config_manager = ConfigManager()
    data_preprocessor = DataPreprocessor(logger_handler, config_manager)
    
    # Path to your data file
    data_file_path = r"C:\TheTradingRobotPlug\data\alpha_vantage\tsla_data.csv"
    
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

if __name__ == "__main__":
    main()
