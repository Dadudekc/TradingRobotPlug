import unittest
import tkinter as tk
from tkinter import ttk, messagebox
from unittest.mock import patch, MagicMock
import threading
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.fetcher_gui import FetcherGUI

class TestFetcherGUI(unittest.TestCase):

    def setUp(self):
        # Create a mock application for testing
        self.root = tk.Tk()
        self.app = FetcherGUI()

    def tearDown(self):
        # Clean up after each test
        self.app.destroy()
        self.root.destroy()

    def test_create_alpha_vantage_tab(self):
        # Simulate notebook initialization within FetcherGUI
        self.app.notebook = ttk.Notebook(self.app)
        self.app.notebook.pack(fill='both', expand=True)

        # Call method to create AlphaVantage tab
        self.app.create_alpha_vantage_tab()

        # Get the first tab (AlphaVantage tab)
        tab = self.app.notebook.nametowidget(self.app.notebook.tabs()[0])

        # Test if widgets are correctly placed on the tab
        self.assertEqual(len(tab.winfo_children()), 4)  # Adjust based on actual widget count

    def test_fetch_alpha_data(self):
        # Mock the behavior of alpha_tickers_entry
        self.app.alpha_tickers_entry = MagicMock()
        self.app.alpha_tickers_entry.get.return_value = "AAPL,GOOGL"

        # Mock threading.Thread to avoid actual thread creation
        mock_thread = MagicMock()
        threading.Thread = MagicMock(return_value=mock_thread)

        # Call fetch_alpha_data and assert expected behavior
        self.app.fetch_alpha_data()
        mock_thread.start.assert_called_once()

    # Add more test cases for other methods as needed

if __name__ == '__main__':
    unittest.main()
