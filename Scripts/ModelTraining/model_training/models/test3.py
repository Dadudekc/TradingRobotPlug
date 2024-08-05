import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
import threading
import pandas as pd
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from model_training_main import train_arima, train_advanced_lstm, train_linear_regression, train_neural_network, train_random_forest
from Scripts.Utilities.model_training_utils import LoggerHandler, DataLoader, DataPreprocessor

class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, config_file, scaler_options):
        super().__init__(parent)
        self.config_file = config_file
        self.scaler_options = scaler_options
        self.queue = queue.Queue()
        self.logger_handler = LoggerHandler(self.log_text)
        self.data_loader = DataLoader(self.logger_handler)
        self.data_preprocessor = DataPreprocessor(self.logger_handler)
        self.is_debug_mode = False
        self.log_text = tk.Text(self, height=10, state='disabled')

        self.scaler_type_var = tk.StringVar(self)
        
        self.setup_gui()

    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        self.display_message(f"Debug mode {'enabled' if self.is_debug_mode else 'disabled'}", level="DEBUG")

    def setup_gui(self):
        self.setup_title_label()
        self.setup_data_file_path_section()
        self.setup_scaler_type_selection()
        self.setup_model_type_selection()
        self.setup_training_configurations()
        self.setup_start_training_button()
        self.setup_progress_and_logging()
        self.setup_debug_mode_toggle()
        self.after(100, self.process_queue)

    def setup_title_label(self):
        tk.Label(self, text="Model Training", font=("Helvetica", 16)).pack(pady=10)

    def setup_data_file_path_section(self):
        tk.Label(self, text="Data File Path:").pack()
        self.data_file_entry = tk.Entry(self)
        self.data_file_entry.pack()
        ttk.Button(self, text="Browse", command=self.browse_data_file).pack(pady=5)

    def setup_scaler_type_selection(self):
        tk.Label(self, text="Select Scaler Type:").pack()
        scaler_type_dropdown = ttk.Combobox(self, textvariable=self.scaler_type_var, values=self.scaler_options)
        scaler_type_dropdown.pack()
        scaler_type_dropdown.current(0)

    def setup_model_type_selection(self):
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar(self)
        model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var, 
                                           values=["linear_regression", "random_forest", "neural_network", "LSTM", "ARIMA"])
        model_type_dropdown.pack()
        self.dynamic_options_frame = tk.Frame(self)
        self.dynamic_options_frame.pack(pady=5)

    def setup_training_configurations(self):
        tk.Label(self, text="Training Configurations", font=("Helvetica", 14)).pack(pady=5)
        self.settings_frame = tk.Frame(self)
        self.settings_frame.pack()

    def setup_start_training_button(self):
        self.start_training_button = ttk.Button(self, text="Start Training", command=self.initiate_training)
        self.start_training_button.pack(pady=10)

    def setup_progress_and_logging(self):
        self.progress_var = tk.IntVar(self, value=0)
        ttk.Progressbar(self, variable=self.progress_var, maximum=100).pack(pady=5)
        self.log_text.pack()

    def setup_debug_mode_toggle(self):
        self.debug_button = tk.Button(self, text="Enable Debug Mode", command=self.toggle_debug_mode)
        self.debug_button.pack(pady=5)

    def browse_data_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, file_path)
            self.preview_selected_data(file_path)

    def initiate_training(self):
        data_file = self.data_file_entry.get()
        model_type = self.model_type_var.get()
        scaler_type = self.scaler_type_var.get()

        if not data_file:
            messagebox.showerror("Error", "Data file not selected")
            return

        if not model_type:
            messagebox.showerror("Error", "Model type not selected")
            return

        threading.Thread(target=self.start_training, args=(data_file, model_type, scaler_type)).start()

    def start_training(self, data_file, model_type, scaler_type):
        try:
            if model_type == "ARIMA":
                train_arima(symbol="AAPL", threshold=100)
            elif model_type == "LSTM":
                train_advanced_lstm(data_file_path=data_file)
            elif model_type == "linear_regression":
                train_linear_regression(data_file_path=data_file)
            elif model_type == "neural_network":
                train_neural_network(data_file_path=data_file, model_config_name="dense_model")
            elif model_type == "random_forest":
                train_random_forest(data_file_path=data_file)
            else:
                self.display_message(f"Unknown model type: {model_type}", level="ERROR")
        except Exception as e:
            self.display_message(f"Error during training: {str(e)}", level="ERROR")

    def process_queue(self):
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self.logger_handler.log(message)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def display_message(self, message, level="INFO"):
        if level == "DEBUG":
            self.logger_handler.debug(message)
        elif level == "ERROR":
            self.logger_handler.error(message)
        else:
            self.logger_handler.info(message)

    def preview_selected_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            self.display_message("Data preview:\n" + str(data.head()), level="INFO")
        except Exception as e:
            self.display_message(f"Error loading data: {str(e)}", level="ERROR")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model Training Application")
    app = ModelTrainingTab(root, config_file='path_to_your_config_file.ini', scaler_options=["StandardScaler", "MinMaxScaler"])
    app.pack(expand=True, fill="both")
    root.mainloop()
