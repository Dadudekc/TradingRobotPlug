import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from datetime import datetime

# Adjust the path to include the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

# Import necessary classes from the Model_Training directory
from Scripts.ModelTraining.DataHandler import DataHandler
from Model_Training.ModelTrainer import ModelTrainer
from Model_Training.ModelEvaluator import ModelEvaluator
from Model_Training.HyperparameterTuner import HyperparameterTuner
from Model_Training.AutomatedModelTrainer import AutomatedModelTrainer
from Model_Training.ModelTrainer import ModelTrainingLogger

# Helper function to create tooltips
def create_tooltip(widget, text):
    tooltip = tk.Label(widget, text=text, background="yellow", wraplength=200)
    def on_enter(event):
        tooltip.place(x=event.x + 20, y=event.y)
    def on_leave(event):
        tooltip.place_forget()
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

# Class for the main application
class Application(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.title("Automated Model Trainer")
        self.geometry("800x600")
        self.config = config

        self.create_widgets()

    def create_widgets(self):
        # Logging text widget
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.log_text = tk.Text(log_frame, state='disabled', height=10, width=80)
        self.log_text.pack(fill="both", expand=True)
        
        # Schedule dropdown
        schedule_label = ttk.Label(self, text="Schedule:")
        schedule_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')

        self.schedule_dropdown = ttk.Combobox(self, values=["Daily", "Weekly", "Monthly"])
        self.schedule_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        self.schedule_dropdown.current(0)
        create_tooltip(self.schedule_dropdown, "Select the schedule for automated training")

        # Start button
        start_button = ttk.Button(self, text="Start Automated Training", command=self.start_automated_training)
        start_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def log_message(self, message, level="INFO"):
        log_colors = {"INFO": "black", "WARNING": "orange", "ERROR": "red", "DEBUG": "blue"}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp} - {level}] {message}\n"
        self.log_text.config(state='normal')
        self.log_text.tag_config(level, foreground=log_colors.get(level, "black"))
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_automated_training(self):
        try:
            # Example of how to start the automated training
            self.log_message("Starting automated training...")
            trainer = AutomatedModelTrainer(self.config, self.schedule_dropdown, self.log_text, data_handler, model_trainer, model_evaluator, hyperparameter_tuner)
            trainer.start_automated_training()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

# Example configuration
config = {
    "Data": {"file_path": "path/to/data.csv", "y_test": np.array([1, 2, 3]), "y_pred": np.array([1.1, 1.9, 3.2])},
    "Model": {"model_type": "neural_network", "epochs": "10", "param_distributions": {"units": [32, 64, 128], "dropout": [0.1, 0.2, 0.3]}}
}

# Initialize components (assuming they are defined elsewhere)
data_handler = DataHandler(config, None)
model_trainer = ModelTrainer(config, None)
model_evaluator = ModelEvaluator(None)
hyperparameter_tuner = HyperparameterTuner(logger=ModelTrainingLogger(None))

# Start the application
app = Application(config)
app.mainloop()
