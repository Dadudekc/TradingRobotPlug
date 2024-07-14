import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Data_Fetchers.data_fetch_main import main as fetch_data_main
from Scripts.Utilities.data_store import DataStore

class DataFetchTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Ticker Symbols (comma-separated):").grid(row=0, column=0, padx=10, pady=10)
        self.symbols_entry = ttk.Entry(self)
        self.symbols_entry.grid(row=0, column=1, padx=10, pady=10)

        default_end_date = datetime.now().strftime('%Y-%m-%d')
        default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        ttk.Label(self, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=10)
        self.start_date_entry = ttk.Entry(self)
        self.start_date_entry.insert(0, default_start_date)
        self.start_date_entry.grid(row=1, column=1, padx=10, pady=10)
        self.start_date_entry.bind("<FocusIn>", lambda event: self.clear_default_date(event, self.start_date_entry))

        ttk.Label(self, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=10)
        self.end_date_entry = ttk.Entry(self)
        self.end_date_entry.insert(0, default_end_date)
        self.end_date_entry.grid(row=2, column=1, padx=10, pady=10)
        self.end_date_entry.bind("<FocusIn>", lambda event: self.clear_default_date(event, self.end_date_entry))

        self.fetch_button = ttk.Button(self, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=3, column=0, columnspan=2, pady=20)

        self.all_data_button = ttk.Button(self, text="Fetch All Data", command=self.fetch_all_data)
        self.all_data_button.grid(row=4, column=0, columnspan=2, pady=20)

        self.display_button = ttk.Button(self, text="Display Chart", command=self.display_chart)
        self.display_button.grid(row=5, column=0, columnspan=2, pady=20)

        self.status_label = ttk.Label(self, text="")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)

    def clear_default_date(self, event, entry):
        entry.delete(0, tk.END)

    def fetch_data(self):
        symbols = self.symbols_entry.get().strip().split(',')
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        # Validate the dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            self.status_label.config(text="Invalid date format. Please use YYYY-MM-DD.")
            return

        # Run the async data fetch function
        asyncio.run(self.run_fetch_data(symbols, start_date, end_date))

    async def run_fetch_data(self, symbols, start_date, end_date):
        self.status_label.config(text="Fetching data...")
        fetched_files = await fetch_data_main(symbols, start_date, end_date)
        if fetched_files:
            self.status_label.config(text=f"Data fetched and saved: {', '.join(fetched_files)}")
        else:
            self.status_label.config(text="No data fetched.")

    def fetch_all_data(self):
        # Fetch all available data without date filters
        symbols = self.symbols_entry.get().strip().split(',')

        # Using a very broad date range to fetch all data
        start_date = "1900-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')

        asyncio.run(self.run_fetch_data(symbols, start_date, end_date))

    def display_chart(self):
        symbols = self.symbols_entry.get().strip().split(',')

        if not symbols:
            self.status_label.config(text="No symbols provided.")
            return

        data_store = DataStore()
        for symbol in symbols:
            data = data_store.load_data(symbol)
            if data is not None:
                self.plot_candlestick_chart(data, symbol)
            else:
                self.status_label.config(text=f"No data found for symbol: {symbol}")

    def plot_candlestick_chart(self, data, symbol):
        df = pd.DataFrame(data)

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ))

        fig.update_layout(title=f'Candlestick Chart for {symbol}', xaxis_title='Date', yaxis_title='Price')
        fig.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataFetchTab(root)
    app.pack(expand=True, fill='both')
    root.mainloop()
