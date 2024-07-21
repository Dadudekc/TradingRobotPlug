import os
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
import lime

# Adjust the Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.hyperparameter_utils import HyperParameterUtils
from Scripts.ModelTraining.hyper_parameter.feature_engineering import FeatureEngineering
from Scripts.Utilities.DataHandler import DataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuning:
    def __init__(self, model, param_grid, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_trials=100, n_jobs=-1, early_stopping=True):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.cv = cv
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.early_stopping = early_stopping

    def objective(self, trial):
        try:
            params = {
                key: trial.suggest_categorical(key, value) if isinstance(value[0], str) else (
                    trial.suggest_int(key, value[0], value[1]) if isinstance(value[0], int) else
                    trial.suggest_float(key, value[0], value[1])
                )
                for key, value in self.param_grid.items()
            }
            self.model.set_params(**params)

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.model.fit(X_train_split, y_train_split)

            y_pred = self.model.predict(X_val_split)
            score = mean_squared_error(y_val_split, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric('mean_squared_error', score)
            mlflow.log_metric('r2_score', r2_score(y_val_split, y_pred))
            mlflow.log_metric('mean_absolute_error', mean_absolute_error(y_val_split, y_pred))

            return score
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            return float('inf')

    def perform_hyperparameter_tuning(self):
        logger.info("Starting advanced hyperparameter tuning with Optuna and MLflow...")

        mlflow.start_run()

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score achieved: {best_score}")

        mlflow.log_params(best_params)
        mlflow.log_metric('best_score', best_score)

        self.model.set_params(**best_params)
        self.model.fit(self.X_train, self.y_train)

        df = study.trials_dataframe()
        df.to_csv("hyperparameter_tuning_results.csv", index=False)
        logger.info("Hyperparameter tuning results saved to hyperparameter_tuning_results.csv")

        fig_optimization = plot_optimization_history(study)
        fig_importance = plot_param_importances(study)
        fig_optimization.show()
        fig_importance.show()

        fig_optimization.write_html("optimization_history.html")
        fig_importance.write_html("param_importances.html")

        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_train)
        shap.summary_plot(shap_values, self.X_train, show=False)
        plt.savefig("shap_summary_plot.png")
        logger.info("SHAP summary plot saved to shap_summary_plot.png")

        for i in range(self.X_train.shape[1]):
            shap.dependence_plot(i, shap_values, self.X_train, show=False)
            plt.savefig(f"shap_dependence_plot_feature_{i}.png")
            logger.info(f"SHAP dependence plot for feature {i} saved to shap_dependence_plot_feature_{i}.png")

        mlflow.end_run()
        return self.model

# Main script
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    # Define the configuration file path
    config_file = 'path_to_your_config_file'  # Update this path to your actual configuration file

    # Initialize DataHandler
    data_handler = DataHandler(config_file=config_file)

    # Load your dataset
    data_path = r'C:\TheTradingRobotPlug\data\alpha_vantage\tsla_data.csv'
    data = data_handler.load_data(data_path)

    # Print the first few rows to inspect
    if data is not None:
        print("Initial data preview:")
        print(data.head())

        # Preprocess the data
        X_train, X_val, y_train, y_val = data_handler.preprocess_data(data, target_column='close')

        if X_train is not None and y_train is not None:
            # Define your model and parameter grid
            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],  # Avoiding None
            }

            # Hyperparameter Tuning
            ht = HyperparameterTuning(model, param_grid, X_train, y_train)

            # Debugging: Check the parameters before starting tuning
            print("Starting hyperparameter tuning with the following parameter grid:")
            print(param_grid)

            # Perform hyperparameter tuning
            best_model = ht.perform_hyperparameter_tuning()

            # Print the best model parameters
            print(f"Best Model Parameters: {best_model.get_params()}")

        else:
            logger.error("Data preprocessing failed. Hyperparameter tuning aborted.")
    else:
        logger.error("Data loading failed. Hyperparameter tuning aborted.")
