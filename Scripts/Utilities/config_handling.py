# C:\TheTradingRobotPlug\Scripts\Utilities\config_handling.py

import configparser
import os

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = self.load_config(config_file)

    def load_config(self, config_file='config.ini'):
        """
        Loads configuration from a .ini file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            configparser.ConfigParser: ConfigParser object containing the configuration.
        """
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        
        config.read(config_file)
        return config

    def get_paths(self):
        """
        Loads paths from the configuration file.

        Returns:
            dict: Dictionary with paths.
        """
        if 'Paths' in self.config:
            paths = {
                'data_folder': self.config.get('Paths', 'data_folder', fallback='/default/data/path'),
                'loading_path': self.config.get('Paths', 'loading_path', fallback='/default/loading/path'),
                'saving_path': self.config.get('Paths', 'saving_path', fallback='/default/saving/path')
            }
            return paths
        else:
            return {
                'data_folder': '/default/data/path',
                'loading_path': '/default/loading/path',
                'saving_path': '/default/saving/path'
            }

    def get_user_settings(self):
        """
        Loads user settings from the configuration file.

        Returns:
            dict: Dictionary with user settings.
        """
        if 'UserSettings' in self.config:
            return dict(self.config['UserSettings'])
        else:
            return {}
