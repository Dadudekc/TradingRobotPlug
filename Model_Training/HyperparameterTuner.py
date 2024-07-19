# C:\TheTradingRobotPlug\Model_Training\HyperparameterTuner.py

import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparameterTuner:
    def __init__(self, logger=None):
        self.logger = logger

    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")

    def perform_hyperparameter_tuning(self, model, X_train, y_train, param_distributions, n_iter=100, cv=3, scoring='neg_mean_squared_error', random_state=42):
        self.log("Starting hyperparameter tuning...", "INFO")
        randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, n_jobs=-1)
        randomized_search.fit(X_train, y_train)
        self.log("Hyperparameter tuning completed.", "INFO")
        return randomized_search.best_estimator_, randomized_search.best_params_

    def create_ensemble_model(self, base_models, train_data, train_labels, method='voting', weights=None):
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            if isinstance(base_models[0], Classifier):
                ensemble_model = VotingClassifier(estimators=base_models, voting='hard', weights=weights)
            else:
                ensemble_model = VotingRegressor(estimators=base_models, weights=weights)
        elif method == 'stacking':
            # Implement stacking ensemble method
            pass
        else:
            raise ValueError("Unsupported ensemble method.")

        ensemble_model.fit(train_data, train_labels)
        self.log(f"Ensemble model created using method {method}.", "INFO")
        return ensemble_model

    def quantize_model(self, model, quantization_method='weight'):
        if quantization_method == 'weight':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()
        elif quantization_method == 'activation':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_generator
            quantized_model = converter.convert()
        else:
            raise ValueError("Unsupported quantization method.")
        
        self.log(f"Model quantized using method {quantization_method}.", "INFO")
        return quantized_model

    def representative_dataset_generator(self, train_data):
        dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(1)
        for input_data in dataset.take(100):
            yield [input_data]

    def initialize_and_configure_model(self, model_type, input_shape, epochs):
        if model_type == "neural_network":
            model = Sequential([
                Dense(128, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        elif model_type == "LSTM":
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        elif model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "ARIMA":
            return {'order': (5, 1, 0)}
        else:
            raise ValueError("Unsupported model type")

        self.log(f"Model of type {model_type} initialized and configured.", "INFO")
        return model

    def create_windowed_data(self, X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)


# Example usage:
# logger = ...  # Initialize your logger here
# tuner = HyperparameterTuner(logger)
# model = tuner.initialize_and_configure_model('neural_network', (10,), 50)
# best_model, best_params = tuner.perform_hyperparameter_tuning(model, X_train, y_train, param_distributions)
