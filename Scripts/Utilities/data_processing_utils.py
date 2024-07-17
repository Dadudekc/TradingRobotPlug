import pandas as pd

class DataValidation:
    @staticmethod
    def validate_dataframe(df, required_columns):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
