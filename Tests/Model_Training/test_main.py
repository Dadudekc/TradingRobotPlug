import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from main_script import main  # Ensure this matches the name of your script file
from gui_module import ModelTrainingTab

class TestMainApplication(unittest.TestCase):

    @patch('main_script.ModelTrainingTab')
    @patch('main_script.tk.Tk')
    def test_main(self, mock_tk, mock_model_training_tab):
        # Create a mock root window and a mock ModelTrainingTab instance
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_model_training_instance = MagicMock()
        mock_model_training_tab.return_value = mock_model_training_instance
        
        # Call the main function
        main()
        
        # Assert that the Tk root window was created and titled
        mock_tk.assert_called_once()
        mock_root.title.assert_called_once_with("Model Training Application")
        
        # Assert that the ModelTrainingTab was created with the correct parameters
        expected_config = {"Paths": {"models_directory": "./models"}}
        expected_scaler_options = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"]
        mock_model_training_tab.assert_called_once_with(mock_root, expected_config, expected_scaler_options)
        
        # Assert that the ModelTrainingTab instance was packed and the main loop was started
        mock_model_training_instance.pack.assert_called_once_with(expand=True, fill="both")
        mock_root.mainloop.assert_called_once()

if __name__ == "__main__":
    unittest.main()
