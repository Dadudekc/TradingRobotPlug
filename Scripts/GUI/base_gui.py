# C:\TheTradingRobotPlug\Scripts\Data_Fetch\base_gui.py

import tkinter as tk
from tkinter import ttk

class BaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Robot GUI")
        self.root.geometry("800x600")

        # Setup main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Setup tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add default tabs
        self.setup_tabs()

    def setup_tabs(self):
        self.add_tab("Home", self.create_home_tab)
        # Additional tabs can be added here by default

    def add_tab(self, title, create_tab_func):
        tab = ttk.Frame(self.tab_control)
        self.tab_control.add(tab, text=title)
        create_tab_func(tab)

    def create_home_tab(self, tab):
        ttk.Label(tab, text="Welcome to the Trading Robot GUI!").grid(row=0, column=0, padx=10, pady=10)
