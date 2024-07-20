import unittest
from unittest.mock import MagicMock, patch, call
import tkinter as tk
from tkinter import ttk
import queue
import os
import sys

# Add project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.DataHandler import DataHandler  # Ensure this matches your module name
from model_training import ModelTraining
from logging_module import ModelTrainingLogger
from utilities import MLRobotUtils

from model_training_tab import ModelTrainingTab  # Ensure this matches your module name

class TestModelTrainingTab(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.title("Test Model Training Application")
        self.config = {"key": "value"}
        self.scaler_options = ["StandardScaler", "MinMaxScaler"]
        self.app = ModelTrainingTab(self.root, self.config, self.scaler_options)

    def tearDown(self):
        self.app.destroy()
        self.root.destroy()

    def test_initialization(self):
        self.assertEqual(self.app.config, self.config)
        self.assertEqual(self.app.scaler_options, self.scaler_options)
        self.assertIsInstance(self.app.queue, queue.Queue)
        self.assertIsInstance(self.app.utils, MLRobotUtils)
        self.assertIsInstance(self.app.logger, ModelTrainingLogger)
        self.assertIsInstance(self.app.model_training, ModelTraining)
        self.assertIsInstance(self.app.data_handler, DataHandler)

    def test_toggle_debug_mode(self):
        initial_state = self.app.is_debug_mode
        self.app.toggle_debug_mode()
        self.assertNotEqual(self.app.is_debug_mode, initial_state)

    @patch('tkinter.filedialog.askopenfilename', return_value="test.csv")
    def test_browse_data_file(self, mock_askopenfilename):
        self.app.browse_data_file()
        self.assertEqual(self.app.data_file_entry.get(), "test.csv")

    def test_start_training_no_data_file(self):
        self.app.data_file_entry.delete(0, tk.END)
        with self.assertRaises(ValueError) as context:
            self.app.start_training()
        self.assertEqual(str(context.exception), "Data file not selected")

    def test_start_training_no_model_type(self):
        self.app.data_file_entry.insert(0, "test.csv")
        self.app.model_type_var.set("")
        with self.assertRaises(ValueError) as context:
            self.app.start_training()
        self.assertEqual(str(context.exception), "Model type not selected")

    @patch.object(DataHandler, 'preprocess_data', return_value=(None, None, None, None))
    def test_start_training_preprocess_data_failure(self, mock_preprocess_data):
        self.app.data_file_entry.insert(0, "test.csv")
        self.app.model_type_var.set("linear_regression")
        self.app.scaler_type_var.set("StandardScaler")
        self.app.epochs_entry = tk.Entry(self.app)
        self.app.epochs_entry.insert(0, "50")
        self.app.start_training()
        mock_preprocess_data.assert_called_once_with("test.csv", "StandardScaler", "linear_regression")

    @patch.object(ModelTraining, 'start_training')
    @patch.object(DataHandler, 'preprocess_data', return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    def test_start_training_success(self, mock_preprocess_data, mock_start_training):
        self.app.data_file_entry.insert(0, "test.csv")
        self.app.model_type_var.set("linear_regression")
        self.app.scaler_type_var.set("StandardScaler")
        self.app.epochs_entry = tk.Entry(self.app)
        self.app.epochs_entry.insert(0, "50")
        self.app.start_training()
        mock_preprocess_data.assert_called_once_with("test.csv", "StandardScaler", "linear_regression")
        mock_start_training.assert_called_once()

    def test_process_queue(self):
        self.app.queue.put("Test message")
        self.app.process_queue()
        self.assertEqual(self.app.log_text.get("1.0", tk.END).strip(), "Test message")

    def test_display_message_info(self):
        self.app.display_message("Info message", level="INFO")
        self.assertIn("Info message", self.app.log_text.get("1.0", tk.END))

    def test_display_message_debug(self):
        self.app.display_message("Debug message", level="DEBUG")
        self.assertIn("Debug message", self.app.log_text.get("1.0", tk.END))

    def test_display_message_error(self):
        self.app.display_message("Error message", level="ERROR")
        self.assertIn("Error message", self.app.log_text.get("1.0", tk.END))

if __name__ == "__main__":
    unittest.main()
