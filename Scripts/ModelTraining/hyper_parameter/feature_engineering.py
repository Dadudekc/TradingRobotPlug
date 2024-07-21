import featuretools as ft
import pandas as pd
import logging
from typing import List, Tuple, Optional

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame, target_column: str, index_column: str = 'index'):
        self.df = df
        self.target_column = target_column
        self.index_column = index_column
        self.setup_logging()
        self.validate_data()

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Logging is set up.")

    def validate_data(self) -> None:
        logging.info("Validating input data...")
        if self.index_column not in self.df.columns:
            logging.error(f"Index column '{self.index_column}' not found in dataframe.")
            raise ValueError(f"Index column '{self.index_column}' not found in dataframe.")
        if not pd.api.types.is_integer_dtype(self.df[self.index_column]):
            logging.error(f"Index column '{self.index_column}' must be of integer type.")
            raise ValueError(f"Index column '{self.index_column}' must be of integer type.")
        if self.target_column not in self.df.columns:
            logging.error(f"Target column '{self.target_column}' not found in dataframe.")
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
        logging.info("Data validation successful.")

    def automated_feature_engineering(
        self, 
        agg_primitives: Optional[List[str]] = None, 
        trans_primitives: Optional[List[str]] = None, 
        max_depth: int = 2
    ) -> Tuple[pd.DataFrame, List[ft.Feature]]:
        if agg_primitives is None:
            agg_primitives = ['mean', 'sum', 'count', 'max', 'min']
        if trans_primitives is None:
            trans_primitives = ['day', 'year', 'month', 'weekday', 'cum_sum', 'cum_mean']

        try:
            logging.info("Starting automated feature engineering...")
            es = ft.EntitySet(id='data')
            es = es.add_dataframe(dataframe_name='main', dataframe=self.df, index=self.index_column)
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name='main',
                agg_primitives=agg_primitives,
                trans_primitives=trans_primitives,
                max_depth=max_depth
            )

            logging.info("Feature engineering completed successfully.")
            return feature_matrix, feature_defs

        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            raise

    def save_feature_matrix(self, feature_matrix: pd.DataFrame, filename: str) -> None:
        try:
            feature_matrix.to_csv(filename, index=False)
            logging.info(f"Feature matrix saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save feature matrix: {e}")
            raise

    def load_feature_matrix(self, filename: str) -> pd.DataFrame:
        try:
            feature_matrix = pd.read_csv(filename)
            logging.info(f"Feature matrix loaded from {filename}")
            return feature_matrix
        except IOError as e:
            logging.error(f"Failed to load feature matrix: {e}")
            raise
