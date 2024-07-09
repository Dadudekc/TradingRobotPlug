# C:\TheTradingRobotPlug\Tests\Utilities\test_config_handling.py

import unittest
import os
import configparser
from Scripts.Utilities.config_handling import ConfigManager

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.test_config_file = 'test_config.ini'
        self.config_manager = ConfigManager(config_file=self.test_config_file)

    def tearDown(self):
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)

    def test_initialization(self):
        self.assertEqual(self.config_manager.config_file, self.test_config_file)
        self.assertTrue(isinstance(self.config_manager.config, configparser.ConfigParser))

    def test_get_set(self):
        self.config_manager.set('section1', 'option1', 'value1')
        value = self.config_manager.get('section1', 'option1')
        self.assertEqual(value, 'value1')

    def test_get_with_fallback(self):
        value = self.config_manager.get('nonexistent_section', 'nonexistent_option', fallback='default_value')
        self.assertEqual(value, 'default_value')

    def test_save(self):
        self.config_manager.set('section1', 'option1', 'value1')
        self.config_manager.save()
        self.config_manager.config.read(self.test_config_file)
        value = self.config_manager.config.get('section1', 'option1')
        self.assertEqual(value, 'value1')

    def test_set_without_section(self):
        self.config_manager.set('new_section', 'new_option', 'new_value')
        value = self.config_manager.get('new_section', 'new_option')
        self.assertEqual(value, 'new_value')

if __name__ == '__main__':
    unittest.main()
