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

from Scrap.model_training_main2 import ModelTraining  # Adjusted import to match your script
from logging_module import ModelTrainingLogger  # Assuming this is a custom logger module

class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, config_file, scaler_options):
        super().__init__(parent)
        self.config_file = config_file
        self.scaler_options = scaler_options
        self.queue = queue.Queue()
        self.is_debug_mode = False
        self.log_text = tk.Text(self, height=10, state='disabled')
        self.logger = ModelTrainingLogger(self.log_text)
        self.model_training = ModelTraining(self.logger)

        self.scaler_type_var = tk.StringVar(self)

        self.setup_gui()

    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        btn_text = "Disable Debug Mode" if self.is_debug_mode else "Enable Debug Mode"
        self.debug_button.config(text=btn_text)
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
        model_type_dropdown.bind("<<ComboboxSelected>>", self.show_dynamic_options)
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

    def show_dynamic_options(self, *_):
        for widget in self.dynamic_options_frame.winfo_children():
            widget.destroy()

        selected_model_type = self.model_type_var.get()
        if hasattr(self.model_training, 'get_model_config'):
            config = self.model_training.get_model_config(selected_model_type)
            if config:
                for key, value in config.items():
                    tk.Label(self.dynamic_options_frame, text=key).pack()
                    entry = tk.Entry(self.dynamic_options_frame)
                    entry.insert(0, str(value))
                    entry.pack()
                    setattr(self, f"{key}_entry", entry)

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

        # Start training in a separate thread
        threading.Thread(target=self.run_training, args=(data_file, model_type, scaler_type)).start()

    def run_training(self, data_file, model_type, scaler_type):
        self.model_training.run_training(data_file, model_type, scaler_type, self.queue)

    def process_queue(self):
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self.display_message(message)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def display_message(self, message, level="INFO"):
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)

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
