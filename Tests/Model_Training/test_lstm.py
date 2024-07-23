import numpy as np
from lime import lime_tabular
from sklearn.linear_model import LinearRegression

# Generate some synthetic data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_val = np.random.rand(20, 10)

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_wrapper(input_data):
    return model.predict(input_data).flatten()

# Create the LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, mode='regression')

# Explain a single instance
exp = explainer.explain_instance(X_val[0], predict_wrapper, num_features=10)

# Display the explanation
exp.show_in_notebook()
