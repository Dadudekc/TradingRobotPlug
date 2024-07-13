import tkinter as tk
from tkinter import ttk
from data_fetch_tab import DataFetchTab

class BaseApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Trading Robot Application")
        self.geometry("800x600")

        # Set the Azure theme
        self.tk.call("source", "Scripts/GUI/azure.tcl")
        self.tk.call("set_theme", "light")

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
