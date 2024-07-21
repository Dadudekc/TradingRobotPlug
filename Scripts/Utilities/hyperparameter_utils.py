# C:\TheTradingRobotPlug\Scripts\Utilities\hyperparameter_utils.py

import json
import os
import logging
from typing import Dict, Any
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import HyperbandPruner
import joblib
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Adjust the Python path dynamically for independent execution
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sys.path.append(str(project_root))

from Scripts.Utilities.config_handling import ConfigManager
from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.DataHandler import DataHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperParameterUtils:
    def __init__(self, config_file='config.ini', data_store: DataStore = None, data_handler: DataHandler = None):
        # Initialize ConfigManager
        defaults = {
            'HYPERPARAMETER_OPTIMIZATION': {
                'n_trials': 100,
                'direction': 'minimize'
            }
        }
        self.config_manager = ConfigManager(config_file=config_file, defaults=defaults)

        self.data_store = data_store
        self.data_handler = data_handler
        self.config = self.config_manager.config

    @staticmethod
    def save_hyperparameters(params: Dict[str, Any], filename: str) -> None:
        if not filename.endswith('.json'):
            logging.error("Filename must end with .json")
            raise ValueError("Filename must end with .json")

        try:
            with open(filename, 'w') as f:
                json.dump(params, f, indent=4)
            logging.info(f"Hyperparameters saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save hyperparameters: {e}")
            raise

    @staticmethod
    def load_hyperparameters(filename: str) -> Dict[str, Any]:
        if not os.path.exists(filename):
            logging.error(f"File {filename} does not exist")
            raise FileNotFoundError(f"File {filename} does not exist")

        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            logging.info(f"Hyperparameters loaded from {filename}")
            return params
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load hyperparameters: {e}")
            raise

    @staticmethod
    def validate_hyperparameters(params: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        for key, value in schema.items():
            if key not in params:
                logging.error(f"Missing hyperparameter: {key}")
                return False
            if not isinstance(params[key], value):
                logging.error(f"Invalid type for hyperparameter: {key}, expected {value}")
                return False
        logging.info("All hyperparameters are valid")
        return True

    def optimize_hyperparameters(
        self,
        symbol: str,
        search_space: Dict[str, Any],
        target_column: str = 'close',
        n_trials: int = None,
        direction: str = None,
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    ) -> Dict[str, Any]:
        logging.info("Starting hyperparameter optimization")

        if n_trials is None:
            n_trials = self.config_manager.get('HYPERPARAMETER_OPTIMIZATION', 'n_trials', 100)
        if direction is None:
            direction = self.config_manager.get('HYPERPARAMETER_OPTIMIZATION', 'direction', 'minimize')

        X_train, X_val, y_train, y_val = self.data_handler.fetch_and_preprocess_data(
            symbol, target_column=target_column
        )

        if X_train is None:
            logging.error(f"Data preprocessing failed for symbol {symbol}")
            return {}

        def objective(trial):
            params = {
                key: trial.suggest_categorical(key, value) if isinstance(value[0], str)
                else trial.suggest_float(key, value[0], value[1]) if isinstance(value, list) and len(value) == 2
                else trial.suggest_int(key, value[0], value[1]) if isinstance(value, list) and len(value) == 2
                else trial.suggest_loguniform(key, value[0], value[1])
                for key, value in search_space.items()
            }
            # Example training and evaluation function
            return self.train_and_evaluate(params, X_train, X_val, y_train, y_val)

        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        logging.info(f"Optimization completed. Best parameters: {best_params}")
        return best_params

    @staticmethod
    def train_and_evaluate(params: Dict[str, Any], X_train, X_val, y_train, y_val) -> float:
        # Example model and evaluation
        model = RandomForestClassifier()  # Replace with actual model initialization
        try:
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = accuracy_score(y_val, predictions)
            return score
        except Exception as e:
            logging.error(f"Error during model training/evaluation: {e}")
            return float('inf')  # Return a large number to indicate failure

    @staticmethod
    def save_study(study: optuna.Study, filename: str) -> None:
        if not filename.endswith('.pkl'):
            logging.error("Filename must end with .pkl")
            raise ValueError("Filename must end with .pkl")

        try:
            with open(filename, 'wb') as f:
                joblib.dump(study, f)
            logging.info(f"Study saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save study: {e}")
            raise

    @staticmethod
    def load_study(filename: str) -> optuna.Study:
        if not os.path.exists(filename):
            logging.error(f"File {filename} does not exist")
            raise FileNotFoundError(f"File {filename} does not exist")

        try:
            with open(filename, 'rb') as f:
                study = joblib.load(f)
            logging.info(f"Study loaded from {filename}")
            return study
        except (IOError, joblib.externals.loky.process_executor._RemoteTraceback) as e:
            logging.error(f"Failed to load study: {e}")
            raise

    @staticmethod
    def generate_report(study: optuna.Study, filename: str = 'hyperparameter_optimization_report.txt') -> None:
        with open(filename, 'w') as f:
            f.write(f"Best Parameters:\n{json.dumps(study.best_params, indent=4)}\n\n")
            f.write(f"Study Statistics:\nNumber of trials: {len(study.trials)}\n")
            f.write(f"Best trial value: {study.best_value}\n")
            f.write(f"Best trial params: {study.best_params}\n")
            f.write("\nTrials Details:\n")
            for trial in study.trials:
                f.write(f"Trial {trial.number}:\n")
                f.write(f"  Value: {trial.value}\n")
                f.write(f"  Params: {trial.params}\n")
                f.write(f"  State: {trial.state}\n\n")
        logging.info(f"Report generated: {filename}")

# Example usage
if __name__ == "__main__":
    config_file = 'config.ini'
    data_store = DataStore()
    data_handler = DataHandler(config_file=config_file, data_store=data_store)
    
    hp_utils = HyperParameterUtils(config_file=config_file, data_store=data_store, data_handler=data_handler)
    search_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    best_params = hp_utils.optimize_hyperparameters('AAPL', search_space)
    print("Best Parameters:", best_params)
