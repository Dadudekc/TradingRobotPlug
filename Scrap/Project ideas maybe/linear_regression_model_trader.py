import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from tpot import TPOTRegressor
from kafka import KafkaConsumer
import json
import mlflow
import mlflow.sklearn
import logging
from concurrent.futures import ThreadPoolExecutor
import joblib
import os
from cryptography.fernet import Fernet

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Setup MLflow
mlflow.set_tracking_uri('http://localhost:5000')  # Update with your server address
mlflow.set_experiment('fintech_model_training')

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def anonymize_data(data):
    """Anonymize sensitive data."""
    if 'user_id' in data.columns:
        data['user_id'] = data['user_id'].apply(lambda x: cipher_suite.encrypt(x.encode()).decode())
    if 'age' in data.columns:
        data['age'] = data['age'].apply(lambda x: f"{x//10*10}-{x//10*10+9}")
    return data

def decrypt_data(data):
    """Decrypt pseudonymized data for internal use if necessary."""
    if 'user_id' in data.columns:
        data['user_id'] = data['user_id'].apply(lambda x: cipher_suite.decrypt(x.encode()).decode())
    return data

def train_linear_regression(X_train, y_train, X_val, y_val, logger=None):
    scaler = StandardScaler()
    selector = SelectFromModel(estimator=Ridge(alpha=1.0), threshold='mean')
    param_grid = {'ridge__alpha': np.logspace(-4, 0, 100)}
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selection', selector),
        ('ridge', Ridge())
    ])
    
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
        randomized_search.fit(X_train, y_train)
    except Exception as e:
        if logger:
            logger.error(f"Error during model training: {str(e)}")
        return None
    
    if logger:
        logger.info("Randomized Search Results:")
        results_df = pd.DataFrame(randomized_search.cv_results_)
        results_str = results_df[['param_ridge__alpha', 'mean_test_score', 'std_test_score']].to_string()
        logger.info(results_str)

    cv_scores = cross_val_score(randomized_search.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    if logger:
        cv_scores_str = ", ".join([f"{score:.2f}" for score in cv_scores])
        logger.info(f"CV Scores: {cv_scores_str}")

    best_model = randomized_search.best_estimator_
    y_pred_val = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    if logger:
        logger.info(f"Validation Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        best_alpha = randomized_search.best_params_['ridge__alpha']
        logger.info(f"Best regularization strength (alpha): {best_alpha:.4f}. Consider using this as a starting point for your next training session.")
    
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_param("best_alpha", best_alpha)
    
    return best_model

def explain_model_with_shap(model, X_train, X_val, logger=None):
    explainer = shap.Explainer(model.named_steps['ridge'], X_train)
    shap_values = explainer(X_val)
    
    if logger:
        logger.info("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_val)
    
    if logger:
        logger.info("SHAP Dependence Plot:")
    shap.dependence_plot(0, shap_values, X_val)

def explain_model_with_lime(model, X_train, X_val, logger=None):
    explainer = LimeTabularExplainer(X_train.values, mode='regression', feature_names=X_train.columns)
    for i in range(len(X_val)):
        exp = explainer.explain_instance(X_val.iloc[i].values, model.predict, num_features=5)
        if logger:
            logger.info(f"LIME Explanation for instance {i}:")
        exp.show_in_notebook(show_all=False)

def train_linear_regression_with_explainability(X_train, y_train, X_val, y_val, logger=None):
    best_model = train_linear_regression(X_train, y_train, X_val, y_val, logger)
    if best_model is not None:
        explain_model_with_shap(best_model, X_train, X_val, logger)
        explain_model_with_lime(best_model, X_train, X_val, logger)
    return best_model

def consume_streaming_data(topic, model, logger=None):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        data = pd.DataFrame([message.value])
        data = anonymize_data(data)  # Anonymize data before prediction
        prediction = model.predict(data)
        if logger:
            logger.info(f"Real-time Prediction: {prediction}")

def train_with_automl(X_train, y_train, X_val, y_val, logger=None):
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, cv=5)
    
    try:
        tpot.fit(X_train, y_train)
    except Exception as e:
        if logger:
            logger.error(f"Error during TPOT training: {str(e)}")
        return None
    
    y_pred_val = tpot.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    r2 = r2_score(y_val, y_pred_val)
    
    if logger:
        logger.info(f"Validation Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    return tpot.fitted_pipeline_

def deploy_model(model, model_name, version):
    """Deploy model using joblib and Docker."""
    model_filename = f"{model_name}_v{version}.pkl"
    joblib.dump(model, model_filename)
    
    dockerfile_content = f"""
    FROM python:3.8-slim
    COPY {model_filename} /app/
    WORKDIR /app
    RUN pip install scikit-learn joblib
    CMD ["python", "-c", "import joblib; model = joblib.load('{model_filename}'); print('Model loaded successfully')"]
    """
    
    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    os.system(f"docker build -t {model_name}_v{version} .")
    os.system(f"docker tag {model_name}_v{version} your_dockerhub_username/{model_name}_v{version}")
    os.system(f"docker push your_dockerhub_username/{model_name}_v{version}")

with mlflow.start_run():
    X_train, y_train, X_val, y_val = load_data()  # Define your data loading function
    X_train = anonymize_data(X_train)
    X_val = anonymize_data(X_val)
    
    best_model = train_linear_regression_with_explainability(X_train, y_train, X_val, y_val, logger)
    consume_streaming_data('your_topic', best_model, logger)
    auto_model = train_with_automl(X_train, y_train, X_val, y_val, logger)
    deploy_model(best_model, "linear_regression_model", "1.0")
