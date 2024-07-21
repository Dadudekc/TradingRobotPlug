import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler

# Adjust the Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.ModelTraining.hyper_parameter.feature_engineering import FeatureEngineering
from Scripts.ModelTraining.hyper_parameter.hyperparameter_tuning import HyperparameterTuning
from Scripts.Utilities.hyperparameter_utils import Utils

# Load your dataset
data_path = r'C:\TheTradingRobotPlug\data\alpha_vantage\tsla_data.csv'
data = pd.read_csv(data_path)

# Print the first few rows to inspect
print("Initial data preview:")
print(data.head())

# Ensure the 'index' column is present or create one
if 'index' not in data.columns:
    data.reset_index(drop=True, inplace=True)
    data['index'] = data.index

# Drop columns with too many NaN values
data.dropna(axis=1, thresh=int(0.8 * len(data)), inplace=True)  # Keeping columns with at least 80% non-NaN values

# Convert 'date' column to datetime format if applicable
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Fill remaining NaNs for numeric columns only
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Print the first few rows to verify the filling
print("Data after filling missing values:")
print(data.head())

# Label Encoding for non-numeric columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'target':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Split the data into training and test sets
X = data.drop(columns=['close'])
y = data['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering
fe = FeatureEngineering(data, target_column='close')
feature_matrix, feature_defs = fe.automated_feature_engineering()

X_train = feature_matrix.loc[X_train.index].drop(columns=['close'])
y_train = feature_matrix.loc[X_train.index]['close']
X_test = feature_matrix.loc[X_test.index].drop(columns=['close'])
y_test = feature_matrix.loc[X_test.index]['close']

# Define your model and parameter grid
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
}

# Hyperparameter Tuning
ht = HyperparameterTuning(model, param_grid, X_train, y_train)

# Debugging: Check the parameters before starting tuning
print("Starting hyperparameter tuning with the following parameter grid:")
print(param_grid)

# Create a new study
study = optuna.create_study(direction='minimize', sampler=TPESampler())

# Define the objective function
def objective(trial):
    n_estimators = trial.suggest_categorical('n_estimators', param_grid['n_estimators'])
    max_depth = trial.suggest_categorical('max_depth', param_grid['max_depth'])
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    return -score  # Minimize the negative score to maximize R^2

# Perform the optimization
study.optimize(objective, n_trials=100, n_jobs=-1)

# Get the best trial
best_trial = study.best_trial
print(f"Best trial: {best_trial.number}")
print(f"Best value: {best_trial.value}")
print(f"Best parameters: {best_trial.params}")

# Train the best model on the training set
best_model = RandomForestRegressor(**best_trial.params)
best_model.fit(X_train, y_train)

# Validate the model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test R^2: {r2}")

print(f"Best Model Parameters: {best_model.get_params()}")
