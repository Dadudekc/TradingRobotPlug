# C:\TheTradingRobotPlug\Tests\GUI\test_fetcher_gui.py

import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.fetcher_gui import FetcherGUI  # Ensure this path is correct

class TestFetcherGUI(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = FetcherGUI(self.root)

    def tearDown(self):
        self.root.destroy()

    @patch('Scripts.GUI.fetcher_gui.messagebox.showinfo')
    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.alpha_vantage_fetcher.AlphaVantageDataFetcher.fetch_data')
    def test_alpha_vantage_fetch_success(self, mock_fetch_data, mock_showerror, mock_showinfo):
        mock_fetch_data.return_value = {'AAPL': MagicMock()}
        self.app.alpha_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_alpha_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showinfo.assert_called_with("Success", "Data fetched for: AAPL")
        mock_showerror.assert_not_called()

    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.alpha_vantage_fetcher.AlphaVantageDataFetcher.fetch_data')
    def test_alpha_vantage_fetch_failure(self, mock_fetch_data, mock_showerror):
        mock_fetch_data.side_effect = Exception("Fetch error")
        self.app.alpha_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_alpha_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showerror.assert_called_with("Error", "Fetch error")

    @patch('Scripts.GUI.fetcher_gui.messagebox.showwarning')
    def test_alpha_vantage_fetch_no_input(self, mock_showwarning):
        self.app.alpha_tickers_entry.delete(0, tk.END)
        self.app.fetch_alpha_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showwarning.assert_called_with("Input Error", "Please enter ticker symbols.")

    @patch('Scripts.GUI.fetcher_gui.messagebox.showinfo')
    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.nasdaq_fetcher.NasdaqDataFetcher.fetch_data')
    def test_nasdaq_fetch_success(self, mock_fetch_data, mock_showerror, mock_showinfo):
        mock_fetch_data.return_value = {'AAPL': MagicMock()}
        self.app.nasdaq_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_nasdaq_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showinfo.assert_called_with("Success", "Data fetched for: AAPL")
        mock_showerror.assert_not_called()

    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.nasdaq_fetcher.NasdaqDataFetcher.fetch_data')
    def test_nasdaq_fetch_failure(self, mock_fetch_data, mock_showerror):
        mock_fetch_data.side_effect = Exception("Fetch error")
        self.app.nasdaq_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_nasdaq_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showerror.assert_called_with("Error", "Fetch error")

    @patch('Scripts.GUI.fetcher_gui.messagebox.showwarning')
    def test_nasdaq_fetch_no_input(self, mock_showwarning):
        self.app.nasdaq_tickers_entry.delete(0, tk.END)
        self.app.fetch_nasdaq_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showwarning.assert_called_with("Input Error", "Please enter ticker symbols.")

    @patch('Scripts.GUI.fetcher_gui.messagebox.showinfo')
    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.polygon_fetcher.PolygonDataFetcher.fetch_data')
    def test_polygon_fetch_success(self, mock_fetch_data, mock_showerror, mock_showinfo):
        mock_fetch_data.return_value = {'AAPL': MagicMock()}
        self.app.polygon_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_polygon_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showinfo.assert_called_with("Success", "Data fetched for: AAPL")
        mock_showerror.assert_not_called()

    @patch('Scripts.GUI.fetcher_gui.messagebox.showerror')
    @patch('Scripts.DataFetchers.polygon_fetcher.PolygonDataFetcher.fetch_data')
    def test_polygon_fetch_failure(self, mock_fetch_data, mock_showerror):
        mock_fetch_data.side_effect = Exception("Fetch error")
        self.app.polygon_tickers_entry.insert(0, 'AAPL')
        self.app.fetch_polygon_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showerror.assert_called_with("Error", "Fetch error")

    @patch('Scripts.GUI.fetcher_gui.messagebox.showwarning')
    def test_polygon_fetch_no_input(self, mock_showwarning):
        self.app.polygon_tickers_entry.delete(0, tk.END)
        self.app.fetch_polygon_data()
        self.app.root.update_idletasks()
        self.app.root.update()
        mock_showwarning.assert_called_with("Input Error", "Please enter ticker symbols.")

if __name__ == '__main__':
    unittest.main()

