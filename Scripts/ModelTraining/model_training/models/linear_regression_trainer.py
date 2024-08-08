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

    def explain_with_lime(self, X_train, y_train, X_val, y_val, strategies=None):
        if strategies is None:
            strategies = ["worst", "best", "median", "25th_quantile", "75th_quantile", 
                        "high_variance", "cluster_center", "outlier", "boundary_case", 
                        "high_influence", "feature_extreme", "new_data", "cluster_error"]

        X_train_selected = self.selector.transform(X_train)
        X_val_selected = self.selector.transform(X_val)

        for strategy in strategies:
                # Generate predictions to compute errors
                predictions = self.best_model.predict(X_val_selected)
                errors = np.abs(predictions - y_val)

                # Determine the index of the instance to explain based on the chosen strategy
                if strategy == "worst":
                    index = np.argmax(errors)
                elif strategy == "best":
                    index = np.argmin(errors)
                elif strategy == "median":
                    median_error = np.median(errors)
                    index = np.argmin(np.abs(errors - median_error))
                elif strategy == "25th_quantile":
                    quantile_error = np.percentile(errors, 25)
                    index = np.argmin(np.abs(errors - quantile_error))
                elif strategy == "75th_quantile":
                    quantile_error = np.percentile(errors, 75)
                    index = np.argmin(np.abs(errors - quantile_error))
                elif strategy == "high_variance":
                    variances = np.var([self.best_model.predict(X_val_selected) for _ in range(10)], axis=0)
                    index = np.argmax(variances)
                elif strategy == "cluster_center":
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=5, random_state=42)
                    kmeans.fit(X_val_selected)
                    centers = kmeans.cluster_centers_
                    index = np.argmin(np.sum((X_val_selected - centers[:, np.newaxis])**2, axis=2).min(axis=1))
                elif strategy == "outlier":
                    outlier_threshold = 3 * np.std(errors)
                    outliers = np.where(errors > outlier_threshold)[0]
                    if outliers.size > 0:
                        index = outliers[0]
                    else:
                        self.logger.info(f"No significant outliers found for strategy '{strategy}'. Defaulting to 'worst' error.")
                        index = np.argmax(errors)
                elif strategy == "boundary_case":
                    boundary_threshold = 0.05  # Example threshold, adjust as needed
                    boundary_cases = np.where((predictions < boundary_threshold) | (predictions > (1 - boundary_threshold)))[0]
                    if boundary_cases.size > 0:
                        index = boundary_cases[0]
                    else:
                        self.logger.info(f"No significant boundary cases found for strategy '{strategy}'. Defaulting to 'worst' error.")
                        index = np.argmax(errors)
                elif strategy == "high_influence":
                    from sklearn.linear_model import LinearRegression
                    influence_model = LinearRegression()
                    influence_model.fit(X_train_selected, y_train)
                    influence_scores = np.abs(influence_model.coef_)
                    index = np.argmax(influence_scores)
                elif strategy == "feature_extreme":
                    extreme_threshold = 0.01  # Example threshold for extremes, adjust as needed
                    extremes = np.where((X_val_selected < extreme_threshold) | (X_val_selected > (1 - extreme_threshold)))[0]
                    if extremes.size > 0:
                        index = extremes[0]
                    else:
                        self.logger.info(f"No significant feature extremes found for strategy '{strategy}'. Defaulting to 'worst' error.")
                        index = np.argmax(errors)
                elif strategy == "new_data":
                    # Identify instances that are far from the training data (e.g., using distance metrics or clustering)
                    from sklearn.neighbors import NearestNeighbors
                    neigh = NearestNeighbors(n_neighbors=1)
                    neigh.fit(X_train_selected)
                    distances, _ = neigh.kneighbors(X_val_selected)
                    index = np.argmax(distances)
                elif strategy == "cluster_error":
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=5, random_state=42)
                    kmeans.fit(X_val_selected)
                    cluster_labels = kmeans.labels_
                    cluster_errors = [errors[cluster_labels == i].mean() for i in range(5)]
                    worst_cluster = np.argmax(cluster_errors)
                    cluster_indices = np.where(cluster_labels == worst_cluster)[0]
                    index = cluster_indices[np.argmax(errors.iloc[cluster_indices])]  # Use iloc for positional indexing


                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                # Create a LIME explainer
                explainer = LimeTabularExplainer(X_train_selected, mode='regression')
                
                # Explain the selected instance
                exp = explainer.explain_instance(X_val_selected[index], self.best_model.predict, num_features=5)
                
                if self.logger:
                    self.logger.info(f"LIME Explanation for instance with {strategy} error (index {index}) saved as lime_explanation_{strategy}.html")
                
                # Save the explanation file in the 'lime_explanations' directory
                explanation_path = os.path.join(lime_explanation_dir, f'lime_explanation_{strategy}.html')
                exp.save_to_file(explanation_path)

    def train_with_explainability(self, X_train, y_train, X_val, y_val):
        self.train(X_train, y_train, X_val, y_val)
        if self.best_model is not None:
            self.explain_with_shap(X_train, X_val)
            self.explain_with_lime(X_train, y_train, X_val, y_val)  # Corrected to pass y_train and y_val here
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
        rmse = np.sqrt(mse)  # Updated to use the standard method of calculating RMSE
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
