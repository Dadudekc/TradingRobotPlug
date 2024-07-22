# first line: 20
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
