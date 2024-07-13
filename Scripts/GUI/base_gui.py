import os
import sys
import tkinter as tk
from tkinter import ttk

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.GUI.data_fetch_tab import DataFetchTab

class BaseApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Trading Robot Application")
        self.geometry("800x600")

        # Commented out Azure theme configuration for now
        # azure_theme_path = os.path.join(script_dir, "azure.tcl")
        # if os.path.exists(azure_theme_path):
        #     self.tk.call("source", azure_theme_path)
        #     self.tk.call("set_theme", "light")
        # else:
        #     print(f"Azure theme file not found at: {azure_theme_path}")

        self.create_widgets()

    def create_widgets(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each tab
        self.create_tabs()

    def create_tabs(self):
        # Adding Data Fetch tab
        self.data_fetch_tab = DataFetchTab(self.notebook)
        self.notebook.add(self.data_fetch_tab, text='Data Fetch')

        # Example additional tabs
        self.backtest_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_tab, text='Backtest')

        self.model_train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_train_tab, text='Model Train')

        self.deploy_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.deploy_tab, text='Deploy')

if __name__ == "__main__":
    app = BaseApp()
    app.mainloop()
