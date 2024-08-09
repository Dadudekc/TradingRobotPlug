import openai
import os
from openai import ChatCompletion

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIUtils:
    def __init__(self, model="gpt-4"):
        self.model = model

    def suggest_new_features(self, data_columns):
        """Suggest new derived features based on existing columns."""
        prompt = f"Given the following financial indicators: {data_columns}, suggest any new derived features or transformations that could improve a trading model's predictions."
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=150)
        return response.choices[0].text.strip()

    def suggest_hyperparameters(self, model_name):
        """Suggest hyperparameters for a given model."""
        prompt = f"I am using a {model_name} model to predict stock prices. What are some optimal hyperparameters I should consider tuning for better performance?"
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=150)
        return response.choices[0].text.strip()

    def generate_model_report(self, mse, rmse, mae, r2):
        """Generate a report on model performance with improvement suggestions."""
        prompt = (f"The model has an MSE of {mse:.2f}, RMSE of {rmse:.2f}, MAE of {mae:.2f}, and R² of {r2:.2f}. "
                  "Given these metrics, what improvements could be made to the model, and how can the results be interpreted?")
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=200)
        return response.choices[0].text.strip()

    def suggest_anomaly_detection_methods(self):
        """Suggest methods for detecting anomalies in financial time series data."""
        prompt = "What are some effective methods to detect and handle anomalies in financial time series data?"
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=150)
        return response.choices[0].text.strip()

    def suggest_trading_strategy(self):
        """Suggest trading strategies based on time series data."""
        prompt = "Given a time series data set of stock prices, what are some potential trading strategies that could be profitable?"
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=200)
        return response.choices[0].text.strip()

    def risk_management_suggestions(self):
        """Provide suggestions for managing and mitigating trading risks."""
        prompt = "Given a trading model with certain risk factors, how can I manage or mitigate these risks effectively?"
        response = ChatCompletion.create(model=self.model, prompt=prompt, max_tokens=200)
        return response.choices[0].text.strip()

    def chat_with_user(self, user_query):
        """Interact with the user via a chat-based interface."""
        response = ChatCompletion.create(model=self.model, prompt=user_query, max_tokens=150)
        return response.choices[0].text.strip()
