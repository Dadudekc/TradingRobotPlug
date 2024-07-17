from technical_indicators.trend_indicators import TrendIndicators
from technical_indicators.momentum_indicators import MomentumIndicators
from technical_indicators.volatility_indicators import VolatilityIndicators
from technical_indicators.volume_indicators import VolumeIndicators
from technical_indicators.custom_indicators import CustomIndicators
from utils.data_validation import DataValidation
import pandas as pd
import talib
# Example usage:
df = pd.read_csv('path_to_your_data.csv')

# Validate DataFrame
DataValidation.validate_dataframe(df, ['high', 'low', 'close', 'volume'])

# Add indicators
df = TrendIndicators.add_moving_average(df)
df = MomentumIndicators.add_relative_strength_index(df)
df = VolatilityIndicators.add_bollinger_bands(df)
df = VolumeIndicators.add_on_balance_volume(df)
df = CustomIndicators.add_fibonacci_retracement_levels(df)
