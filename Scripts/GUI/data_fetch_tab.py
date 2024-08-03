import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import time

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.data_fetch_main import main as fetch_data_main
from Scripts.Utilities.data_store import DataStore
from Scripts.Data_Processing.Technical_indicators.custom_indicators import CustomIndicators
from Scripts.Data_Processing.Technical_indicators.momentum_indicators import MomentumIndicators
from Scripts.Data_Processing.Technical_indicators.trend_indicators import TrendIndicators
from Scripts.Data_Processing.Technical_indicators.volatility_indicators import VolatilityIndicators
from Scripts.Data_Processing.Technical_indicators.volume_indicators import VolumeIndicators

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataFetchTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        # Frame for data fetching
        fetch_frame = ttk.LabelFrame(self, text="Fetch Data")
        fetch_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Ticker symbols entry
        ttk.Label(fetch_frame, text="Ticker Symbols (comma-separated):").grid(row=0, column=0, padx=10, pady=10)
        self.symbols_entry = ttk.Entry(fetch_frame)
        self.symbols_entry.grid(row=0, column=1, padx=10, pady=10)

        # Default date range
        default_end_date = datetime.now().strftime('%Y-%m-%d')
        default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Start date entry
        ttk.Label(fetch_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=10)
        self.start_date_entry = ttk.Entry(fetch_frame)
        self.start_date_entry.insert(0, default_start_date)
        self.start_date_entry.grid(row=1, column=1, padx=10, pady=10)

        # End date entry
        ttk.Label(fetch_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=10)
        self.end_date_entry = ttk.Entry(fetch_frame)
        self.end_date_entry.insert(0, default_end_date)
        self.end_date_entry.grid(row=2, column=1, padx=10, pady=10)

        # Fetch data button
        self.fetch_button = ttk.Button(fetch_frame, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Fetch all data button
        self.all_data_button = ttk.Button(fetch_frame, text="Fetch All Data", command=self.fetch_all_data)
        self.all_data_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Status label
        self.status_label = ttk.Label(fetch_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=10)

        # Frame for selecting indicators
        indicator_frame = ttk.LabelFrame(self, text="Select Indicators")
        indicator_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Dictionary for storing indicator options
        self.indicators = {
            "Sample_Custom_Indicator": tk.BooleanVar(),
            "Another_Custom_Indicator": tk.BooleanVar(),
            "Stochastic": tk.BooleanVar(),
            "RSI": tk.BooleanVar(),
            "Williams %R": tk.BooleanVar(),
            "ROC": tk.BooleanVar(),
            "TRIX": tk.BooleanVar(),
            "SMA": tk.BooleanVar(),
            "EMA": tk.BooleanVar(),
            "MACD": tk.BooleanVar(),
            "ADX": tk.BooleanVar(),
            "Ichimoku": tk.BooleanVar(),
            "PSAR": tk.BooleanVar(),
            "Bollinger Bands": tk.BooleanVar(),
            "Standard Deviation": tk.BooleanVar(),
            "Historical Volatility": tk.BooleanVar(),
            "Chandelier Exit": tk.BooleanVar(),
            "Keltner Channel": tk.BooleanVar(),
            "Moving Average Envelope": tk.BooleanVar(),
            "MFI": tk.BooleanVar(),
            "OBV": tk.BooleanVar(),
            "VWAP": tk.BooleanVar(),
            "ADL": tk.BooleanVar(),
            "CMF": tk.BooleanVar(),
            "Volume Oscillator": tk.BooleanVar()
        }

        # Create checkbuttons for indicators
        row = 0
        col = 0
        for ind, var in self.indicators.items():
            chk = ttk.Checkbutton(indicator_frame, text=ind, variable=var)
            chk.grid(row=row, column=col, sticky="w")
            row += 1
            if row == 12:
                row = 0
                col += 1

        # Button to select all indicators
        self.select_all_button = ttk.Button(indicator_frame, text="Select All", command=self.toggle_select_all)
        self.select_all_button.grid(row=13, column=0, columnspan=2, pady=10)

        # Button to display chart
        self.display_button = ttk.Button(indicator_frame, text="Display Chart", command=self.display_chart)
        self.display_button.grid(row=14, column=0, columnspan=2, pady=10)

    def toggle_select_all(self):
        # Toggle selection of all indicators
        current_state = any(var.get() for var in self.indicators.values())
        new_state = not current_state
        for var in self.indicators.values():
            var.set(new_state)

    def fetch_data(self):
        # Fetch data for specified symbols and date range
        symbols = self.symbols_entry.get().strip().split(',')
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        if not self.validate_dates(start_date, end_date):
            self.status_label.config(text="Invalid date format. Please use YYYY-MM-DD.")
            return

        asyncio.run(self.run_fetch_data(symbols, start_date, end_date))

    def validate_dates(self, start_date, end_date):
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    async def run_fetch_data(self, symbols, start_date, end_date):
        # Run data fetching asynchronously
        self.status_label.config(text="Fetching data...")
        try:
            fetched_files = await fetch_data_main(symbols, start_date, end_date)
            if fetched_files:
                self.status_label.config(text=f"Data fetched and saved: {', '.join(fetched_files)}")
                self.apply_indicators_to_fetched_data(fetched_files)
            else:
                self.status_label.config(text="No data fetched.")
        except Exception as e:
            self.status_label.config(text=f"Error fetching data: {e}")
            logger.error(f"Error fetching data: {e}")

    def fetch_all_data(self):
        # Fetch all available data
        symbols = self.symbols_entry.get().strip().split(',')
        start_date = "1900-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        asyncio.run(self.run_fetch_data(symbols, start_date, end_date))

    def apply_indicators_to_fetched_data(self, fetched_files):
        # Apply selected indicators to fetched data
        data_store = DataStore(csv_dir='C:/TheTradingRobotPlug/data/alpha_vantage')
        for file in fetched_files:
            symbol = file.split('_')[0]
            data = data_store.load_data(symbol)
            if data is not None:
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)

                indicator_functions = {
                    "Sample_Custom_Indicator": CustomIndicators.sample_custom_indicator,
                    "Another_Custom_Indicator": CustomIndicators.another_custom_indicator,
                    "Stochastic": MomentumIndicators.add_stochastic_oscillator,
                    "RSI": MomentumIndicators.add_relative_strength_index,
                    "Williams %R": MomentumIndicators.add_williams_r,
                    "ROC": MomentumIndicators.add_rate_of_change,
                    "TRIX": MomentumIndicators.add_trix,
                    "SMA": lambda df: TrendIndicators.add_moving_average(df, ma_type='SMA'),
                    "EMA": lambda df: TrendIndicators.add_moving_average(df, ma_type='EMA'),
                    "MACD": TrendIndicators.calculate_macd_components,
                    "ADX": TrendIndicators.add_adx,
                    "Ichimoku": TrendIndicators.add_ichimoku_cloud,
                    "PSAR": TrendIndicators.add_parabolic_sar,
                    "Bollinger Bands": VolatilityIndicators.add_bollinger_bands,
                    "Standard Deviation": VolatilityIndicators.add_standard_deviation,
                    "Historical Volatility": VolatilityIndicators.add_historical_volatility,
                    "Chandelier Exit": VolatilityIndicators.add_chandelier_exit,
                    "Keltner Channel": VolatilityIndicators.add_keltner_channel,
                    "Moving Average Envelope": VolatilityIndicators.add_moving_average_envelope,
                    "MFI": VolumeIndicators.add_money_flow_index,
                    "OBV": VolumeIndicators.add_on_balance_volume,
                    "VWAP": VolumeIndicators.add_vwap,
                    "ADL": VolumeIndicators.add_accumulation_distribution_line,
                    "CMF": VolumeIndicators.add_chaikin_money_flow,
                    "Volume Oscillator": VolumeIndicators.add_volume_oscillator
                }

                for indicator in self.indicators.keys():
                    if self.indicators[indicator].get():
                        func = indicator_functions.get(indicator)
                        if func:
                            start_time = time.perf_counter()
                            if indicator in ["Sample_Custom_Indicator", "Another_Custom_Indicator"]:
                                window_size = 5 if indicator == "Sample_Custom_Indicator" else 10
                                df = CustomIndicators.add_custom_indicator(df, indicator, func, window_size=window_size)
                            else:
                                df = func(df)
                            elapsed_time = time.perf_counter() - start_time
                            logger.info(f"Successfully added {indicator} in {elapsed_time:.6f} seconds.")
                        else:
                            logger.warning(f"Unknown indicator selected: {indicator}")

                data_store.save_data(df, symbol, processed=True, overwrite=True)
            else:
                logger.warning(f"No data found for symbol: {symbol}")

    def display_chart(self):
        # Display chart with selected indicators
        symbols = self.symbols_entry.get().strip().split(',')
        selected_indicators = [key for key, var in self.indicators.items() if var.get()]

        if not symbols or symbols == ['']:
            self.status_label.config(text="No symbols provided.")
            return

        data_store = DataStore()
        for symbol in symbols:
            data = data_store.load_data(symbol)
            if data is not None:
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=("Candlestick", "Trend Indicators", "Momentum Indicators"))

                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol
                ), row=1, col=1)

                indicator_mapping = {
                    "Sample_Custom_Indicator": "Sample_Custom_Indicator",
                    "Another_Custom_Indicator": "Another_Custom_Indicator",
                    "Stochastic": ["Stochastic", "Stochastic_Signal"],
                    "RSI": "RSI",
                    "Williams %R": "Williams_R",
                    "ROC": "ROC",
                    "TRIX": ["TRIX", "TRIX_signal"],
                    "SMA": "SMA_10",
                    "EMA": "EMA_10",
                    "MACD": ["MACD", "MACD_Signal", "MACD_Hist", "MACD_Hist_Signal"],
                    "ADX": "ADX",
                    "Ichimoku": ["Ichimoku_Conversion_Line", "Ichimoku_Base_Line", "Ichimoku_Leading_Span_A", "Ichimoku_Leading_Span_B", "Ichimoku_Lagging_Span"],
                    "PSAR": "PSAR",
                    "Bollinger Bands": ["Bollinger_High", "Bollinger_Low", "Bollinger_Mid"],
                    "Standard Deviation": "Standard_Deviation",
                    "Historical Volatility": "Historical_Volatility",
                    "Chandelier Exit": "Chandelier_Exit_Long",
                    "Keltner Channel": ["Keltner_Channel_High", "Keltner_Channel_Low", "Keltner_Channel_Mid"],
                    "Moving Average Envelope": ["MAE_Upper", "MAE_Lower"],
                    "MFI": "MFI",
                    "OBV": "OBV",
                    "VWAP": "VWAP",
                    "ADL": "ADL",
                    "CMF": "CMF",
                    "Volume Oscillator": "Volume_Oscillator"
                }

                for indicator in selected_indicators:
                    col_name = indicator_mapping.get(indicator)
                    if col_name:
                        if isinstance(col_name, list):
                            for col in col_name:
                                if col in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df[col],
                                        mode='lines',
                                        name=col
                                    ), row=2 if indicator in ["SMA", "EMA", "MACD", "ADX", "Ichimoku", "PSAR"] else 3, col=1)
                        else:
                            if col_name in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df[col_name],
                                    mode='lines',
                                    name=col_name
                                ), row=2 if indicator in ["SMA", "EMA", "MACD", "ADX", "Ichimoku", "PSAR"] else 3, col=1)

                fig.update_layout(title=f'Candlestick Chart and Indicators for {symbol}', xaxis_title='Date', yaxis_title='Price')
                fig.show()
            else:
                self.status_label.config(text=f"No data found for symbol: {symbol}")
                return

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Financial Data Fetcher and Indicator Application")
    app = DataFetchTab(root)
    app.pack(expand=True, fill='both')
    root.mainloop()
