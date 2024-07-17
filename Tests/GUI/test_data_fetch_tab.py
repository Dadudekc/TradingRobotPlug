import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import pandas as pd

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.data_fetch_tab import DataFetchTab
from Scripts.Data_Fetchers.data_fetch_main import main as fetch_data_main
from Scripts.Utilities.data_store import DataStore
from Scripts.Data_Processing.Technical_indicators.custom_indicators import CustomIndicators, sample_custom_indicator

class TestDataFetchTab(unittest.TestCase):

    @patch('Scripts.GUI.data_fetch_tab.fetch_data_main')
    def setUp(self, mock_fetch_data_main):
        self.root = tk.Tk()
        self.app = DataFetchTab(self.root)
        self.app.pack()

    def tearDown(self):
        self.root.destroy()

    def test_widgets_created(self):
        self.assertIsInstance(self.app.symbols_entry, ttk.Entry)
        self.assertIsInstance(self.app.start_date_entry, ttk.Entry)
        self.assertIsInstance(self.app.end_date_entry, ttk.Entry)
        self.assertIsInstance(self.app.fetch_button, ttk.Button)
        self.assertIsInstance(self.app.all_data_button, ttk.Button)
        self.assertIsInstance(self.app.display_button, ttk.Button)
        self.assertIsInstance(self.app.status_label, ttk.Label)

    def test_clear_default_date(self):
        self.app.start_date_entry.insert(0, "2023-07-17")
        self.app.start_date_entry.event_generate("<FocusIn>")
        self.assertEqual(self.app.start_date_entry.get(), "")

    @patch('Scripts.GUI.data_fetch_tab.asyncio.run', return_value=None)
    def test_fetch_data(self, mock_asyncio_run):
        self.app.symbols_entry.delete(0, tk.END)  # Clear any existing text
        self.app.symbols_entry.insert(0, "AAPL,MSFT")
        self.app.start_date_entry.delete(0, tk.END)  # Clear any existing text
        self.app.start_date_entry.insert(0, "2022-01-01")
        self.app.end_date_entry.delete(0, tk.END)  # Clear any existing text
        self.app.end_date_entry.insert(0, "2023-01-01")
        self.app.fetch_data()
        mock_asyncio_run.assert_called_once()

    def test_validate_dates(self):
        valid_start_date = "2022-01-01"
        valid_end_date = "2023-01-01"
        invalid_date = "2023-13-01"
        self.assertTrue(self.app.validate_dates(valid_start_date, valid_end_date))
        self.assertFalse(self.app.validate_dates(valid_start_date, invalid_date))

    @patch('Scripts.GUI.data_fetch_tab.DataStore')
    def test_display_chart_no_symbols(self, mock_data_store):
        self.app.display_chart()
        self.assertEqual(self.app.status_label.cget("text"), "No symbols provided.")

    @patch('Scripts.GUI.data_fetch_tab.DataStore')
    @patch('Scripts.GUI.data_fetch_tab.pd.DataFrame')
    @patch('Scripts.GUI.data_fetch_tab.make_subplots')
    @patch('Scripts.GUI.data_fetch_tab.go.Candlestick')
    def test_display_chart_with_data(self, mock_candlestick, mock_make_subplots, mock_dataframe, mock_data_store):
        self.app.symbols_entry.delete(0, tk.END)  # Clear any existing text
        self.app.symbols_entry.insert(0, "AAPL")
        mock_data_store.return_value.load_data.return_value = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
        })
        self.app.display_chart()
        mock_data_store.return_value.load_data.assert_called_once_with("AAPL")
        mock_make_subplots.assert_called_once()
        mock_candlestick.assert_called_once()

    @patch('Scripts.Data_Fetchers.data_fetch_main.main')
    async def test_fetch_data_main(self, mock_fetch_data_main):
        mock_fetch_data_main.return_value = ["file1.csv", "file2.csv"]
        self.app.symbols_entry.insert(0, "AAPL,MSFT")
        self.app.start_date_entry.insert(0, "2022-01-01")
        self.app.end_date_entry.insert(0, "2023-01-01")
        await self.app.fetch_data()
        self.assertEqual(self.app.status_label.cget("text"), "Data fetched and saved: file1.csv, file2.csv")
        mock_fetch_data_main.assert_called_once_with(['AAPL', 'MSFT'], '2022-01-01', '2023-01-01')

    @patch('Scripts.Data_Fetchers.data_fetch_main.main')
    async def test_fetch_all_data(self, mock_fetch_data_main):
        mock_fetch_data_main.return_value = ["file1.csv", "file2.csv"]
        self.app.symbols_entry.insert(0, "AAPL,MSFT")
        await self.app.fetch_all_data()
        expected_text = "Data fetched and saved: file1.csv, file2.csv"
        self.assertEqual(self.app.status_label.cget("text"), expected_text)
        mock_fetch_data_main.assert_called_once_with(['AAPL', 'MSFT'], '1900-01-01', datetime.now().strftime('%Y-%m-%d'))

    @patch.object(DataStore, 'load_data')
    def test_display_chart_with_indicator(self, mock_load_data):
        mock_data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'open': pd.Series(range(100)),
            'high': pd.Series(range(100)),
            'low': pd.Series(range(100)),
            'close': pd.Series(range(100)),
            'volume': pd.Series(range(100))
        }
        mock_load_data.return_value = mock_data
        self.app.symbols_entry.insert(0, "AAPL")
        self.app.indicators["Sample_Custom_Indicator"].set(True)
        with patch.object(DataFetchTab, 'plot_candlestick_chart') as mock_plot:
            self.app.display_chart()
            mock_plot.assert_called_once()
            df = mock_plot.call_args[0][0]
            self.assertIn('Sample_Custom_Indicator', df.columns)

    def test_toggle_select_all(self):
        # Initially, all indicators are deselected
        for var in self.app.indicators.values():
            self.assertFalse(var.get())
        # Select all indicators
        self.app.toggle_select_all()
        for var in self.app.indicators.values():
            self.assertTrue(var.get())
        # Deselect all indicators
        self.app.toggle_select_all()
        for var in self.app.indicators.values():
            self.assertFalse(var.get())

if __name__ == "__main__":
    unittest.main()
