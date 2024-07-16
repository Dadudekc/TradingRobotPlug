# C:\TheTradingRobotPlug\Tests\GUI\test_base_gui.py

import os
import sys
import unittest
from unittest.mock import patch
import tkinter as tk

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.base_gui import BaseGUI

class TestBaseGUI(unittest.TestCase):
    @patch('Scripts.GUI.data_fetch_tab.DataFetchTab', new=lambda x: tk.Frame(x))  # Mocking DataFetchTab
    def setUp(self):
        self.root = tk.Tk()
        self.app = BaseGUI()

    def tearDown(self):
        self.root.destroy()

    def test_initial_title(self):
        self.assertEqual(self.app.title(), "Trading Robot Application")

    def test_initial_geometry(self):
        # Force update the geometry of the window
        self.app.update_idletasks()
        # Split the geometry string and compare only the size part
        geometry_size = self.app.geometry().split('+')[0]
        self.assertEqual(geometry_size, "800x600")

    def test_tabs_created(self):
        expected_tabs = ['Data Fetch', 'Backtest', 'Model Train', 'Deploy']
        actual_tabs = [self.app.notebook.tab(i, "text") for i in range(len(expected_tabs))]
        self.assertEqual(expected_tabs, actual_tabs)

if __name__ == "__main__":
    unittest.main()
