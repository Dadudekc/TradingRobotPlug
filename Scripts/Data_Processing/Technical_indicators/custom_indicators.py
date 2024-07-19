import talib
import os
import sys
import pandas as pd
import logging
from typing import Callable, List, Tuple, Dict, Any
from time import time as timer
import joblib

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.config_handling import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomIndicators:
    _cache = {}
    config_manager = ConfigManager(config_file='config.ini')

    @staticmethod
    def file_cache(key: str, function: Callable, *args: Any, **kwargs: Any) -> pd.Series:
        cache_file = f"cache/{key}.pkl"
        if os.path.exists(cache_file):
            logger.info(f"Loading from file cache: {cache_file}")
            return joblib.load(cache_file)
        result = function(*args, **kwargs)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        joblib.dump(result, cache_file)
        return result

    @staticmethod
    def get_cache_key(df: pd.DataFrame, function_name: str, args: Tuple, kwargs: Dict) -> str:
        df_hash = joblib.hash(df.to_string())
        args_hash = joblib.hash((args, frozenset(kwargs.items())))
        return f"{function_name}_{df_hash}_{args_hash}"

    @staticmethod
    def cached_indicator_function(
        df: pd.DataFrame, 
        indicator_function: Callable, 
        *args: Any, 
        cache_strategy: str = 'memory', 
        **kwargs: Any
    ) -> pd.Series:
        cache_key = CustomIndicators.get_cache_key(df, indicator_function.__name__, args, kwargs)
        
        if cache_strategy == 'memory':
            return CustomIndicators.memory_cache(cache_key, indicator_function, df, *args, **kwargs)
        elif cache_strategy == 'file':
            return CustomIndicators.file_cache(cache_key, indicator_function, df, *args, **kwargs)
        else:
            raise ValueError(f"Unknown cache strategy: {cache_strategy}")

    @staticmethod
    def memory_cache(key: str, function: Callable, *args: Any, **kwargs: Any) -> pd.Series:
        if key not in CustomIndicators._cache:
            CustomIndicators._cache[key] = function(*args, **kwargs)
        return CustomIndicators._cache[key]

    @staticmethod
    def add_custom_indicator(df: pd.DataFrame, indicator_name: str, indicator_function: Callable, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not callable(indicator_function):
            raise ValueError("'indicator_function' must be a callable function.")
        if not isinstance(indicator_name, str) or not indicator_name:
            raise ValueError("'indicator_name' must be a non-empty string.")

        logger.info(f"Adding custom indicator '{indicator_name}' to the DataFrame.")
        try:
            start_time = timer()
            indicator_params = CustomIndicators.config_manager.get('INDICATORS', indicator_name, fallback=None)
            if indicator_params:
                indicator_params = eval(indicator_params)
                kwargs.update(indicator_params)
            cache_strategy = CustomIndicators.config_manager.get('CACHE', 'strategy', fallback='memory')
            df[indicator_name] = CustomIndicators.cached_indicator_function(
                df, indicator_function, *args, cache_strategy=cache_strategy, **kwargs
            )
            end_time = timer()
            logger.info(f"Successfully added custom indicator '{indicator_name}' in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error in executing the custom indicator function '{indicator_name}': {e}")
            raise RuntimeError(f"Error in executing the custom indicator function: {e}")

        return df

    @staticmethod
    def add_multiple_custom_indicators(df: pd.DataFrame, indicators: List[Tuple[str, Callable, List[Any], Dict[str, Any]]]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not isinstance(indicators, list):
            raise ValueError("'indicators' must be a list of tuples.")

        logger.info("Adding multiple custom indicators to the DataFrame.")

        for indicator_name, indicator_function, args, kwargs in indicators:
            df = CustomIndicators.add_custom_indicator(df, indicator_name, indicator_function, *args, **kwargs)

        logger.info("Successfully added all custom indicators.")
        return df

    @staticmethod
    def validate_indicator(indicator_function: Callable) -> bool:
        if not callable(indicator_function):
            logger.error("The provided indicator function is not callable.")
            return False

        try:
            sample_df = pd.DataFrame({
                'close': [1, 2, 3, 4, 5],
                'high': [1, 2, 3, 4, 5],
                'low': [1, 2, 3, 4, 5],
                'volume': [1, 2, 3, 4, 5]
            })
            indicator_function(sample_df)
            return True
        except Exception as e:
            logger.error(f"Validation failed for the indicator function: {e}")
            return False

    @staticmethod
    def sample_custom_indicator(df: pd.DataFrame, window_size: int = 10) -> pd.Series:
        return df['close'].rolling(window=window_size).mean()

    @staticmethod
    def another_custom_indicator(df: pd.DataFrame, window_size: int = 5) -> pd.Series:
        return df['close'].rolling(window=window_size).std()

# Example usage of CustomIndicators
if __name__ == "__main__":
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'close': pd.Series(range(100))
    }
    df = pd.DataFrame(data)

    df = CustomIndicators.add_custom_indicator(df, 'Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, window_size=5)
    print(df.head(10))

    indicators = [
        ('Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, [5], {}),
        ('Another_Custom_Indicator', CustomIndicators.another_custom_indicator, [10], {})
    ]
    df = CustomIndicators.add_multiple_custom_indicators(df, indicators)
    print(df.head(10))
