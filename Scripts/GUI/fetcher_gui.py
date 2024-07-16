import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# Adjust sys.path to include the directory containing 'Scripts'
sys.path.append(os.path.join(project_root, 'Scripts'))

from GUI.base_gui import BaseGUI
from Data_Fetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher
from Data_Fetchers.polygon_fetcher import PolygonDataFetcher

class FetcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        self.create_alpha_vantage_tab()
        self.create_polygon_tab()

    def create_alpha_vantage_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="AlphaVantage Fetcher")

        ttk.Label(tab, text="AlphaVantage Data Fetcher").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(tab, text="Ticker Symbols (comma separated):").grid(row=1, column=0, padx=10, pady=5)
        self.alpha_tickers_entry = ttk.Entry(tab)
        self.alpha_tickers_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Button(tab, text="Fetch Data", command=self.fetch_alpha_data).grid(row=2, column=0, columnspan=2, pady=10)

    def fetch_alpha_data(self):
        tickers = self.alpha_tickers_entry.get()
        if not tickers:
            messagebox.showwarning("Input Error", "Please enter ticker symbols.")
            return

        tickers_list = [ticker.strip() for ticker in tickers.split(",")]
        threading.Thread(target=self.fetch_alpha_data_thread, args=(tickers_list,)).start()

    def fetch_alpha_data_thread(self, tickers_list):
        fetcher = AlphaVantageDataFetcher()
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        try:
            data = fetcher.fetch_data(tickers_list, start_date, end_date)
            if data:
                messagebox.showinfo("Success", f"Data fetched for: {', '.join(data.keys())}")
            else:
                messagebox.showerror("Error", "Failed to fetch data.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_polygon_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Polygon Fetcher")

        ttk.Label(tab, text="Polygon Data Fetcher").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(tab, text="Ticker Symbols (comma separated):").grid(row=1, column=0, padx=10, pady=5)
        self.polygon_tickers_entry = ttk.Entry(tab)
        self.polygon_tickers_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Button(tab, text="Fetch Data", command=self.fetch_polygon_data).grid(row=2, column=0, columnspan=2, pady=10)

    def fetch_polygon_data(self):
        tickers = self.polygon_tickers_entry.get()
        if not tickers:
            messagebox.showwarning("Input Error", "Please enter ticker symbols.")
            return

        tickers_list = [ticker.strip() for ticker in tickers.split(",")]
        threading.Thread(target=self.fetch_polygon_data_thread, args=(tickers_list,)).start()

    def fetch_polygon_data_thread(self, tickers_list):
        fetcher = PolygonDataFetcher()
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        try:
            data = fetcher.fetch_data(tickers_list, start_date, end_date)
            if data:
                messagebox.showinfo("Success", f"Data fetched for: {', '.join(data.keys())}")
            else:
                messagebox.showerror("Error", "Failed to fetch data.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = FetcherGUI()
    app.mainloop()
