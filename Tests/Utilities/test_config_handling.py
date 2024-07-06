# C:\TheTradingRobotPlug\Tests\Utilities\test_config_handling.py

import unittest
import os
import sys
from pathlib import Path

# Ensure the project root is added to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Adjusted to correctly find the project root
sys.path.append(str(project_root))

from Scripts.Utilities.config_handling import ConfigManager

class TestConfigManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_config_file = 'test_config.ini'
        self.config_manager = ConfigManager(config_file=self.test_config_file)

    def tearDown(self):
        """Clean up test environment after each test."""
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)

    def test_set_and_get(self):
        self.config_manager.set('section1', 'key1', 'value1')
        value = self.config_manager.get('section1', 'key1')
        self.assertEqual(value, 'value1')

    def test_get_with_fallback(self):
        value = self.config_manager.get('section1', 'nonexistent_key', fallback='default_value')
        self.assertEqual(value, 'default_value')

    def test_save(self):
        self.config_manager.set('section1', 'key1', 'value1')
        self.config_manager.save()
        new_config_manager = ConfigManager(config_file=self.test_config_file)
        value = new_config_manager.get('section1', 'key1')
        self.assertEqual(value, 'value1')

if __name__ == '__main__':
    unittest.main()
