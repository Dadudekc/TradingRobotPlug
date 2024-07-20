import unittest
from unittest.mock import MagicMock, patch
import logging
import tkinter as tk
from model_training_logger import ModelTrainingLogger, WidgetHandler  # Ensure this matches your module name

class TestModelTrainingLogger(unittest.TestCase):

    def setUp(self):
        # Initialize a Tkinter root and text widget for testing
        self.root = tk.Tk()
        self.log_widget = tk.Text(self.root)
        self.logger = ModelTrainingLogger(self.log_widget)

    def tearDown(self):
        # Destroy the Tkinter root after each test
        self.root.destroy()

    def test_info_logging(self):
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.info("Info message")
            mock_info.assert_called_once_with("Info message")

    def test_error_logging(self):
        with patch.object(self.logger.logger, 'error') as mock_error:
            self.logger.error("Error message")
            mock_error.assert_called_once_with("Error message")

    def test_debug_logging(self):
        with patch.object(self.logger.logger, 'debug') as mock_debug:
            self.logger.debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")

    def test_widget_logging(self):
        # Log a message and check if it appears in the widget
        self.logger.info("Widget info message")
        log_content = self.log_widget.get("1.0", tk.END).strip()
        self.assertIn("Widget info message", log_content)

    def test_widget_handler_emit(self):
        widget_handler = WidgetHandler(self.log_widget)
        record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg="Test message", args=(), exc_info=None)
        widget_handler.emit(record)
        log_content = self.log_widget.get("1.0", tk.END).strip()
        self.assertIn("Test message", log_content)

if __name__ == "__main__":
    unittest.main()
