# C:\TheTradingRobotPlug\Tests\Utilities\test_config_handling.py

import unittest
from Scripts.Utilities.config_handling import ConfigManager


class TestConfigHandling(unittest.TestCase):
    
    def setUp(self):
        self.config_manager = ConfigManager()  # Initialize ConfigManager with default config

    def test_get_paths(self):
        paths = self.config_manager.get_paths()
        self.assertIsInstance(paths, dict)
        self.assertIn('data_folder', paths)
        self.assertIn('loading_path', paths)
        self.assertIn('saving_path', paths)

    def test_get_user_settings(self):
        user_settings = self.config_manager.get_user_settings()
        self.assertIsInstance(user_settings, dict)

if __name__ == '__main__':
    unittest.main()
