# first line: 73
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
