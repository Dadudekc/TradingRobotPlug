# C:\TheTradingRobotPlug\Scripts\ModelTraining\model_training\models\linear_regression.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from tpot import TPOTRegressor
from kafka import KafkaConsumer
import json

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
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
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
            self.logger.info("SHAP Summary Plot:")
        shap.summary_plot(shap_values.values, X_val_selected)

        if self.logger:
            self.logger.info("SHAP Dependence Plot:")
        shap.dependence_plot(0, shap_values.values, X_val_selected)

    def explain_with_lime(self, X_train, X_val):
        X_train_selected = self.selector.transform(X_train)
        X_val_selected = self.selector.transform(X_val)
        
        explainer = LimeTabularExplainer(X_train_selected, mode='regression')
        for i in range(len(X_val_selected)):
            exp = explainer.explain_instance(X_val_selected[i], self.best_model.predict, num_features=5)
            if self.logger:
                self.logger.info(f"LIME Explanation for instance {i}:")
            exp.show_in_notebook(show_all=False)

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
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        r2 = r2_score(y_val, y_pred_val)

        if self.logger:
            self.logger.info(f"Validation Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return tpot.fitted_pipeline_
