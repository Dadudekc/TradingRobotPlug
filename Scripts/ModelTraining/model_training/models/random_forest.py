import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import optuna
from optuna.samplers import TPESampler
from typing import Optional, Tuple, Any, Dict
import logging
import shap
from joblib import Memory
import os

class RandomForestModel:
    def __init__(self, cache_location: Optional[str] = 'cache', logger: Optional[logging.Logger] = None):
        self.memory = Memory(location=os.path.join(os.getcwd(), cache_location), verbose=0)
        self.logger = logger or logging.getLogger(__name__)
        self.best_rf_model = None
        self.best_params = {}

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 3) -> float:
        """
        Objective function for Optuna to minimize the mean squared error using cross-validation.

        Parameters:
        - trial (optuna.Trial): Optuna trial object.
        - X_train (np.ndarray): Training feature data.
        - y_train (np.ndarray): Training target data.
        - cv_folds (int): Number of cross-validation folds. Defaults to 3.

        Returns:
        - mse (float): Mean squared error.
        """
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6)
        }
        
        rf = RandomForestRegressor(random_state=42, **param_grid)
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
        mse = -np.mean(cv_scores)
        
        return mse

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = None,
              n_trials: int = 50, cv_folds: int = 3, cache_enabled: bool = True) -> Tuple[RandomForestRegressor, Dict[str, Any], float, float, float, float, float]:
        """
        Train a random forest model with adaptive hyperparameter tuning using Optuna.

        Parameters:
        - X (np.ndarray): Feature data.
        - y (np.ndarray): Target data.
        - test_size (float): Proportion of the dataset to include in the validation split. Defaults to 0.2.
        - random_state (Optional[int]): Seed for random number generation. Defaults to None.
        - n_trials (int): Number of trials for Optuna optimization. Defaults to 50.
        - cv_folds (int): Number of cross-validation folds. Defaults to 3.
        - cache_enabled (bool): Enable/disable caching of results. Defaults to True.

        Returns:
        - best_rf_model (RandomForestRegressor): The best trained RandomForestRegressor model.
        - best_params (dict): Best hyperparameters found during Optuna optimization.
        - mse (float): Mean squared error on validation data.
        - rmse (float): Root mean squared error on validation data.
        - mae (float): Mean absolute error on validation data.
        - mape (float): Mean absolute percentage error on validation data.
        - r2 (float): R² score on validation data.
        """
        if not cache_enabled:
            self.memory.clear(warn=False)

        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y should be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have the same number of samples.")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=random_state))
        
        # Use the memory.cache decorated function
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
        """
        Predict using the trained RandomForestRegressor model.

        Parameters:
        - X (np.ndarray): Feature data.

        Returns:
        - predictions (np.ndarray): Predicted values.
        """
        if self.best_rf_model is None:
            raise ValueError("The model has not been trained yet. Please call `train` first.")
        
        return self.best_rf_model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained RandomForestRegressor model.

        Returns:
        - feature_importances (np.ndarray): Feature importances.
        """
        if self.best_rf_model is None:
            raise ValueError("The model has not been trained yet. Please call `train` first.")
        
        return self.best_rf_model.feature_importances_

# Example usage with logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Replace with actual data
    X, y = np.random.rand(100, 10), np.random.rand(100)

    model = RandomForestModel(logger=logger)
    best_model, best_params, mse, rmse, mae, mape, r2 = model.train(X, y, random_state=42)
    logger.info(f"Best model: {best_model}")
