# random_forest_trainer.py
# Location: Scripts/ModelTraining/
# Enhanced for training a Random Forest model with advanced feature engineering and time series cross-validation.

import os
import sys
import logging
import numpy as np
import pandas as pd
import shap
from joblib import Memory
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
import optuna
from optuna.samplers import TPESampler
from typing import Optional, Tuple, Any, Dict

# Set up project root and add 'Utilities' to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]
utilities_dir = project_root / 'Scripts' / 'Utilities'

# Add the Utilities directory to sys.path
if utilities_dir.exists() and str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

# Set up relative paths for resources and logs
resources_path = project_root / 'resources'
log_path = project_root / 'logs'

# Ensure the directories exist
resources_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
log_file = log_path / 'random_forest_trainer.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger('RandomForestTrainer')

class RandomForestModel:
    def __init__(self, cache_location: Optional[str] = 'cache', logger: Optional[logging.Logger] = None):
        cache_path = Path(cache_location)
        self.memory = Memory(location=str(cache_path), verbose=0)
        self.logger = logger or logging.getLogger(__name__)
        self.best_rf_model = None
        self.best_params = {}

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example feature engineering
        df['lag_1'] = df['close'].shift(1)
        df['lag_5'] = df['close'].shift(5)
        df['lag_10'] = df['close'].shift(10)
        df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
        df['rolling_std_5'] = df['close'].rolling(window=5).std()
        df['RSI'] = self.calculate_rsi(df['close'])
        # Drop NaN rows resulting from shifts and rolling operations
        df = df.dropna()
        return df
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 3) -> float:
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6)
        }
        
        rf = RandomForestRegressor(random_state=42, **param_grid)
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        
        # Use TimeSeriesSplit for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)
        mse = -np.mean(cv_scores)
        
        return mse

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = None,
              n_trials: int = 50, cv_folds: int = 5, cache_enabled: bool = True) -> Tuple[RandomForestRegressor, Dict[str, Any], float, float, float, float, float]:
        if not cache_enabled:
            self.memory.clear(warn=False)

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y should be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have the same number of samples.")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)  # Avoid shuffling for time series

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=random_state))
        cached_objective = self.memory.cache(self.objective)
        study.optimize(lambda trial: cached_objective(trial, X_train, y_train, cv_folds), n_trials=n_trials)

        self.best_params = study.best_params
        self.best_rf_model = RandomForestRegressor(random_state=random_state, **self.best_params)
        self.best_rf_model.fit(X_train, y_train)
        
        y_pred_val = self.best_rf_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        mae = mean_absolute_error(y_val, y_pred_val)
        mape = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100
        r2 = self.best_rf_model.score(X_val, y_val)
        
        if self.logger:
            self.logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, R²: {r2:.2f}")
            self.logger.info(f"Best parameters found: {self.best_params}. Use these parameters as a baseline for your next training session.")
        
        # Feature importance
        if self.logger:
            feature_importances = self.best_rf_model.feature_importances_
            self.logger.info(f"Feature importances: {feature_importances}")

        # SHAP values for interpretability
        explainer = shap.TreeExplainer(self.best_rf_model)
        shap_values = explainer.shap_values(X_val)
        if self.logger:
            shap.summary_plot(shap_values, X_val, plot_type="bar")
        
        return self.best_rf_model, self.best_params, mse, rmse, mae, mape, r2

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_rf_model is None:
            raise ValueError("The model has not been trained yet. Please call `train` first.")
        
        return self.best_rf_model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        if self.best_rf_model is None:
            raise ValueError("The model has not been trained yet. Please call `train` first.")
        
        return self.best_rf_model.feature_importances_

# Example usage with logging
if __name__ == "__main__":
    # Replace with the actual path to your data
    data_path = project_root / 'data' / 'alpha_vantage' / 'tsla_data.csv'

    # Load the data (assuming the file contains a target column named 'close')
    data = pd.read_csv(data_path)

    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Extract useful features from the date
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_month'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Drop columns that contain non-numeric data
    data = data.drop(columns=['date', 'symbol'])  # Assuming 'symbol' column contains 'tsla'

    # Perform feature engineering
    model = RandomForestModel(logger=logger)
    data = model.feature_engineering(data)

    # Prepare the features (X) and target (y)
    X = data.drop(columns=['close']).values  # Convert to NumPy array
    y = data['close'].values  # Convert to NumPy array

    best_model, best_params, mse, rmse, mae, mape, r2 = model.train(X, y, random_state=42)
    logger.info(f"Best model: {best_model}")
