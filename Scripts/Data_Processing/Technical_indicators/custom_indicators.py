import talib
import os
import sys
import pandas as pd
import logging
from typing import Callable, List, Tuple, Dict, Any
from time import time as timer  # Avoid conflict with time module
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from functools import wraps

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
        """
        File-based caching using joblib.

        Args:
            key (str): The cache key.
            function (Callable): The function to cache.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            pd.Series: The result of the function.
        """
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
        """
        Generate a unique cache key.

        Args:
            df (pd.DataFrame): DataFrame on which the function is applied.
            function_name (str): Name of the function.
            args (Tuple): Arguments for the function.
            kwargs (Dict): Keyword arguments for the function.

        Returns:
            str: The cache key.
        """
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
        """
        Cached version of the indicator function to avoid recalculating indicators.

        Args:
            df (pd.DataFrame): DataFrame on which the indicator function will be applied.
            indicator_function (Callable): Function that computes the indicator.
            cache_strategy (str): Caching strategy to use ('memory' or 'file').
            *args: Variable length argument list for the indicator function.
            **kwargs: Arbitrary keyword arguments for the indicator function.

        Returns:
            pd.Series: Result of the indicator function.
        """
        cache_key = CustomIndicators.get_cache_key(df, indicator_function.__name__, args, kwargs)
        
        if cache_strategy == 'memory':
            return CustomIndicators.memory_cache(cache_key, indicator_function, df, *args, **kwargs)
        elif cache_strategy == 'file':
            return CustomIndicators.file_cache(cache_key, indicator_function, df, *args, **kwargs)
        else:
            raise ValueError(f"Unknown cache strategy: {cache_strategy}")

    @staticmethod
    def memory_cache(key: str, function: Callable, *args: Any, **kwargs: Any) -> pd.Series:
        """
        In-memory caching using a dictionary.

        Args:
            key (str): The cache key.
            function (Callable): The function to cache.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            pd.Series: The result of the function.
        """
        if key not in CustomIndicators._cache:
            CustomIndicators._cache[key] = function(*args, **kwargs)
        return CustomIndicators._cache[key]

    @staticmethod
    def add_custom_indicator(df: pd.DataFrame, indicator_name: str, indicator_function: Callable, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds a custom indicator to the DataFrame using a user-defined function.

        Args:
            df (pd.DataFrame): DataFrame to which the indicator will be added.
            indicator_name (str): Name of the new indicator column to be added.
            indicator_function (Callable): Function that computes the indicator.
            *args: Variable length argument list for the indicator function.
            **kwargs: Arbitrary keyword arguments for the indicator function.

        Returns:
            pd.DataFrame: Modified DataFrame with the new indicator column added.

        Raises:
            ValueError: If 'df' is not a pandas DataFrame, 'indicator_function' is not callable,
                        or 'indicator_name' is not a non-empty string.
            RuntimeError: If an error occurs while executing the custom indicator function.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not callable(indicator_function):
            raise ValueError("'indicator_function' must be a callable function.")
        if not isinstance(indicator_name, str) or not indicator_name:
            raise ValueError("'indicator_name' must be a non-empty string.")

        logger.info(f"Adding custom indicator '{indicator_name}' to the DataFrame.")
        try:
            start_time = timer()
            # Retrieve configuration settings if available
            indicator_params = CustomIndicators.config_manager.get('INDICATORS', indicator_name, fallback=None)
            if indicator_params:
                indicator_params = eval(indicator_params)  # Convert string representation of dict to actual dict
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
        """
        Adds multiple custom indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which the indicators will be added.
            indicators (list of tuples): Each tuple contains ('indicator_name', indicator_function, args, kwargs).

        Returns:
            pd.DataFrame: Modified DataFrame with the new indicator columns added.

        Raises:
            ValueError: If any element in indicators is not a tuple or contains invalid elements.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not isinstance(indicators, list):
            raise ValueError("'indicators' must be a list of tuples.")

        logger.info("Adding multiple custom indicators to the DataFrame.")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    CustomIndicators.add_custom_indicator, df.copy(), indicator[0], indicator[1], *indicator[2], **indicator[3]
                ): indicator[0] for indicator in indicators
            }

            for future in as_completed(futures):
                indicator_name = futures[future]
                try:
                    df = future.result()
                    logger.info(f"Successfully added indicator '{indicator_name}'.")
                except Exception as e:
                    logger.error(f"Failed to add indicator '{indicator_name}': {e}")
                    raise e

        logger.info("Successfully added all custom indicators.")
        return df

    @staticmethod
    def validate_indicator(indicator_function: Callable) -> bool:
        """
        Validates the custom indicator function to ensure it can be applied to a DataFrame.

        Args:
            indicator_function (Callable): The indicator function to validate.

        Returns:
            bool: True if the function is valid, False otherwise.
        """
        if not callable(indicator_function):
            logger.error("The provided indicator function is not callable.")
            return False

        try:
            # Create a sample DataFrame to test the function
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
        """
        Example custom indicator that calculates the rolling mean of the 'close' column.

        Args:
            df (pd.DataFrame): DataFrame with stock data.
            window_size (int): Window size for the rolling calculation.

        Returns:
            pd.Series: Calculated rolling mean.
        """
        return df['close'].rolling(window=window_size).mean()

    @staticmethod
    def another_custom_indicator(df: pd.DataFrame, window_size: int = 5) -> pd.Series:
        """
        Example custom indicator that calculates the rolling standard deviation of the 'close' column.

        Args:
            df (pd.DataFrame): DataFrame with stock data.
            window_size (int): Window size for the rolling calculation.

        Returns:
            pd.Series: Calculated rolling standard deviation.
        """
        return df['close'].rolling(window=window_size).std()

# Example usage of CustomIndicators
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'date': pd.date_range(start='2022-01-01', periods=100),
        'close': pd.Series(range(100))
    }
    df = pd.DataFrame(data)

    # Add a single custom indicator to the DataFrame
    df = CustomIndicators.add_custom_indicator(df, 'Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, window_size=5)
    print(df.head(10))

    # Add multiple custom indicators to the DataFrame
    indicators = [
        ('Sample_Custom_Indicator', CustomIndicators.sample_custom_indicator, [5], {}),
        ('Another_Custom_Indicator', CustomIndicators.another_custom_indicator, [10], {})
    ]
    df = CustomIndicators.add_multiple_custom_indicators(df, indicators)
    print(df.head(10))
