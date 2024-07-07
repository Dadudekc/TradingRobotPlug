from Scripts.GUI.base_gui import BaseGUI
from tkinter import ttk
from tkinter import messagebox
import threading

class FetcherGUI(BaseGUI):
    def __init__(self, root):
        super().__init__(root)
        self.add_tab("AlphaVantage Fetcher", self.create_alpha_vantage_tab)
        self.add_tab("Nasdaq Fetcher", self.create_nasdaq_tab)
        self.add_tab("Polygon Fetcher", self.create_polygon_tab)

    def create_alpha_vantage_tab(self, tab):
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
        from Scripts.DataFetchers.alpha_vantage_fetcher import AlphaVantageDataFetcher
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

    def create_nasdaq_tab(self, tab):
        ttk.Label(tab, text="Nasdaq Data Fetcher").grid(row=0, column=0, padx=10, pady=10)
        ttk.Label(tab, text="Ticker Symbols (comma separated):").grid(row=1, column=0, padx=10, pady=5)
        self.nasdaq_tickers_entry = ttk.Entry(tab)
        self.nasdaq_tickers_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Button(tab, text="Fetch Data", command=self.fetch_nasdaq_data).grid(row=2, column=0, columnspan=2, pady=10)

    def fetch_nasdaq_data(self):
        tickers = self.nasdaq_tickers_entry.get()
        if not tickers:
            messagebox.showwarning("Input Error", "Please enter ticker symbols.")
            return

        tickers_list = [ticker.strip() for ticker in tickers.split(",")]
        threading.Thread(target=self.fetch_nasdaq_data_thread, args=(tickers_list,)).start()

    def fetch_nasdaq_data_thread(self, tickers_list):
        from Scripts.DataFetchers.nasdaq_fetcher import NasdaqDataFetcher
        fetcher = NasdaqDataFetcher()
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

    def create_polygon_tab(self, tab):
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
        from Scripts.DataFetchers.polygon_fetcher import PolygonDataFetcher
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
