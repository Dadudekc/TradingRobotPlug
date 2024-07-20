import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from hyperparameter_tuning import perform_hyperparameter_tuning

class TestHyperparameterTuning(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y_train = pd.Series(np.random.rand(100))

        # Initialize a Ridge regression model
        self.model = Ridge()

        # Define a parameter grid for hyperparameter tuning
        self.param_grid = {'alpha': [0.1, 1.0, 10.0]}

    @patch('hyperparameter_tuning.GridSearchCV')
    def test_perform_hyperparameter_tuning(self, mock_grid_search_cv):
        # Create a mock instance of GridSearchCV
        mock_grid_search = MagicMock()
        mock_grid_search_cv.return_value = mock_grid_search

        # Simulate the best estimator
        mock_best_model = MagicMock()
        mock_grid_search.best_estimator_ = mock_best_model

        # Call the perform_hyperparameter_tuning function
        best_model = perform_hyperparameter_tuning(self.model, self.param_grid, self.X_train, self.y_train)

        # Assert that GridSearchCV was called with the correct parameters
        mock_grid_search_cv.assert_called_once_with(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=2
        )

        # Assert that fit was called on the GridSearchCV instance
        mock_grid_search.fit.assert_called_once_with(self.X_train, self.y_train)

        # Assert that the best estimator was returned
        self.assertEqual(best_model, mock_best_model)

if __name__ == "__main__":
    unittest.main()
