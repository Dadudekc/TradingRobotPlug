# C:\TheTradingRobotPlug\Scripts\GUI\base_gui.py

import os
import sys
import tkinter as tk
from tkinter import ttk

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.data_fetch_tab import DataFetchTab

class BaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Base GUI")
        
        self.label = tk.Label(root, text="This is the Base GUI")
        self.label.pack()

    def run(self):
        self.root.mainloop()

class BaseGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Trading Robot Application")
        self.geometry("800x600")

        # Azure theme configuration
        self.configure_theme()

        self.create_widgets()

    def configure_theme(self):
        azure_theme_path = os.path.join(script_dir, "azure.tcl")
        if os.path.exists(azure_theme_path):
            try:
                self.tk.call("source", azure_theme_path)
                self.tk.call("set_theme", "light")
            except tk.TclError as e:
                print(f"Error loading Azure theme: {e}")
        else:
            print(f"Azure theme file not found at: {azure_theme_path}")

    def create_widgets(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Create and add tabs
        self.create_tabs()

    def create_tabs(self):
        tabs = {
            'Data Fetch': DataFetchTab,
            'Backtest': ttk.Frame,
            'Model Train': ttk.Frame,
            'Deploy': ttk.Frame,
        }

        for tab_name, tab_class in tabs.items():
            tab = tab_class(self.notebook)
            self.notebook.add(tab, text=tab_name)

if __name__ == "__main__":
    app = BaseGUI()
    app.mainloop()

