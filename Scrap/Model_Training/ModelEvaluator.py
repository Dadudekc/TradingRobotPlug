#

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import joblib
import traceback
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    explained_variance_score, log_loss, max_error,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, precision_recall_fscore_support,
    r2_score, roc_auc_score
)
from datetime import datetime

class ModelEvaluator:
    def __init__(self, log_text_widget=None):
        self.log_text_widget = log_text_widget
        self.log("ModelEvaluator initialized.")

    def log(self, message, level="INFO"):
        if self.log_text_widget:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp} - {level}] {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')
        else:
            print(f"[{level}] {message}")

    def evaluate_model(self, model, X_test, y_test, model_type):
        try:
            y_pred = model.predict(X_test)
            results_message = "Model Evaluation Results:\n"

            if model_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)

                results_message += f"""Accuracy: {accuracy:.2f}
Precision: {precision:.2f}
Recall: {recall:.2f}
F1-Score: {fscore:.2f}
AUC-ROC: {auc_roc:.2f}
Log Loss: {logloss:.2f}\n"""

                self.plot_confusion_matrix(conf_matrix, ['Class 0', 'Class 1'])

            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                accuracy = self.calculate_model_accuracy(model, X_test, y_test)

                results_message += f"""MSE: {mse:.2f}
RMSE: {rmse:.2f}
R2 Score: {r2:.2f}
Accuracy: {accuracy:.2f}%\n"""

            self.log(results_message)

            return results_message

        except Exception as e:
            error_message = f"Error during model evaluation: {str(e)}\n{traceback.format_exc()}"
            self.log(error_message, "ERROR")
            return None

    def calculate_model_accuracy(self, model, X_test, y_test):
        try:
            if hasattr(model, 'score'):
                accuracy = model.score(X_test, y_test)
                return accuracy * 100.0
        except Exception as e:
            self.log(f"Error calculating model accuracy: {str(e)}", "ERROR")
        return 0.0

    def plot_confusion_matrix(self, conf_matrix, class_names, save_path="confusion_matrix.png", show_plot=True):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

        self.log(f"Confusion matrix plot saved to {save_path}.")

    def visualize_training_results(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Model Evaluation Results')
        plt.grid(True)
        plt.show()
        self.log("Model evaluation visualization displayed.")

    def generate_model_reports(self, model, X_test, y_test, y_pred, model_type):
        try:
            if model_type == 'classification':
                classification_rep = classification_report(y_test, y_pred)
                self.log("Classification Report:\n" + classification_rep)

                confusion_mat = confusion_matrix(y_test, y_pred)
                self.plot_confusion_matrix(confusion_mat, ['Class 0', 'Class 1'])

            elif model_type == 'regression':
                regression_metrics = self.calculate_regression_metrics(y_test, y_pred)
                self.log("Regression Metrics:\n" + json.dumps(regression_metrics, indent=4))

                self.generate_regression_visualizations(y_test, y_pred)

        except Exception as e:
            self.log(f"Error generating model reports: {str(e)}", "ERROR")

    def calculate_regression_metrics(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics = {
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'Mean Absolute Error (MAE)': mae,
            'R-squared (R2)': r2,
            'Explained Variance': explained_variance,
            'Max Error': max_err,
            'Mean Absolute Percentage Error (MAPE)': mape
        }
        return metrics

    def generate_regression_visualizations(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.show()

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def save_evaluation_results(self, results, file_path):
        with open(file_path, 'w') as results_file:
            json.dump(results, results_file, indent=4)
        self.log(f"Evaluation results saved to {file_path}.")

    def load_evaluation_results(self, file_path):
        try:
            with open(file_path, 'r') as results_file:
                results = json.load(results_file)
            self.log(f"Evaluation results loaded from {file_path}.")
            return results
        except Exception as e:
            self.log(f"Failed to load evaluation results from {file_path}: {str(e)}", "ERROR")
            return None

# Example usage:
# log_text_widget = ...  # Your Tkinter Text widget for logging (optional)
# evaluator = ModelEvaluator(log_text_widget)
# evaluation_results = evaluator.evaluate_model(trained_model, X_test, y_test, 'regression')
# evaluator.visualize_training_results(y_test, y_pred)
# evaluator.generate_model_reports(trained_model, X_test, y_test, y_pred, 'regression')
