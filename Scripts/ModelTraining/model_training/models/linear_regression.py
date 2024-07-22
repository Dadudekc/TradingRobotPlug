import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

def train_linear_regression(X_train, y_train, X_val, y_val, logger=None):
    """Train a linear regression model with hyperparameter tuning."""
    param_grid = {'alpha': np.logspace(-4, 0, 50)}
    model = Ridge()
    randomized_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', verbose=2)
    
    try:
        randomized_search.fit(X_train, y_train)
    except Exception as e:
        if logger:
            logger.error(f"Error during model training: {str(e)}")
        return None
    
    if logger:
        logger.info("Randomized Search Results:")
        results_df = pd.DataFrame(randomized_search.cv_results_)
        results_str = results_df[['param_alpha', 'mean_test_score', 'std_test_score']].to_string()
        logger.info(results_str)

    cv_scores = cross_val_score(randomized_search.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    if logger:
        cv_scores_str = ", ".join([f"{score:.2f}" for score in cv_scores])
        logger.info(f"CV Scores: {cv_scores_str}")

    best_model = randomized_search.best_estimator_
    y_pred_val = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    r2 = best_model.score(X_val, y_val)
    
    if logger:
        logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        best_alpha = randomized_search.best_params_['alpha']
        logger.info(f"Best regularization strength (alpha): {best_alpha:.4f}. Consider using this as a starting point for your next training session.")
    
    return best_model
