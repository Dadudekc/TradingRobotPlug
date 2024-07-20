import unittest
from unittest.mock import MagicMock
from MLRobotUtils import MLRobotUtils

class TestMLRobotUtils(unittest.TestCase):
    def setUp(self):
        self.utils = MLRobotUtils()
        self.mock_text_widget = MagicMock()
    
    def test_log_message_debug(self):
        message = "This is a debug message"
        is_debug_mode = True
        
        self.utils.log_message(message, self.mock_text_widget, is_debug_mode)
        
        # Check if the message is inserted into the text widget
        self.mock_text_widget.insert.assert_called_with('end', f"{message}\n")
        self.mock_text_widget.see.assert_called_with('end')
        
        # Check if the message is logged as debug
        with self.assertLogs(self.utils.logger, level='DEBUG') as cm:
            self.utils.logger.debug(message)
            self.assertIn('DEBUG', cm.output[0])
    
    def test_log_message_info(self):
        message = "This is an info message"
        is_debug_mode = False
        
        self.utils.log_message(message, self.mock_text_widget, is_debug_mode)
        
        # Check if the message is inserted into the text widget
        self.mock_text_widget.insert.assert_called_with('end', f"{message}\n")
        self.mock_text_widget.see.assert_called_with('end')
        
        # Check if the message is logged as info
        with self.assertLogs(self.utils.logger, level='INFO') as cm:
            self.utils.logger.info(message)
            self.assertIn('INFO', cm.output[0])

if __name__ == '__main__':
    unittest.main()
