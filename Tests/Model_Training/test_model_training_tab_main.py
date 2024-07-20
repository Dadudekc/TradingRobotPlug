# C:\TheTradingRobotPlug\Tests\Model_Training\test_model_training_tab_main.py
import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk

# Mocking the imports for utilities and gui_module
with patch('utilities.MLRobotUtils'), patch('gui_module.ModelTrainingTab'):

    from main import main

    class TestModelTrainingMain(unittest.TestCase):
        
        @patch('main.tk.Tk', autospec=True)
        @patch('main.ModelTrainingTab', autospec=True)
        def test_main(self, MockModelTrainingTab, MockTk):
            # Creating mock instances
            mock_root = MockTk.return_value
            mock_model_training_tab = MockModelTrainingTab.return_value
            
            # Running the main function
            main()

            # Assertions to ensure Tk and ModelTrainingTab were called correctly
            MockTk.assert_called_once_with()
            mock_root.title.assert_called_once_with("Model Training Application")
            MockModelTrainingTab.assert_called_once_with(mock_root, {"Paths": {"models_directory": "./models"}}, ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"])
            mock_model_training_tab.pack.assert_called_once_with(expand=True, fill="both")
            mock_root.mainloop.assert_called_once_with()

    if __name__ == '__main__':
        unittest.main()
gi
