# C:\TheTradingRobotPlug\Scripts\Utilities\config_handling.py

import configparser
import os
import logging

class ConfigManager:
    def __init__(self, config_file='config.ini', defaults=None):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.logger = logging.getLogger(__name__)
        
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.config['DEFAULT'] = defaults if defaults else {}

        self.load_environment_variables()

    def load_environment_variables(self):
        for section in self.config.sections():
            for key in self.config[section]:
                env_var = os.getenv(f"{section.upper()}_{key.upper()}")
                if env_var:
                    self.logger.debug(f"Overriding {section}.{key} with environment variable.")
                    self.config.set(section, key, env_var)

    def get(self, section, option, fallback=None):
        value = self.config.get(section, option, fallback=fallback)
        self.logger.debug(f"Retrieving {section}.{option}: {value}")
        return value

    def set(self, section, option, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
            self.logger.debug(f"Adding section: {section}")
        self.config.set(section, option, value)
        self.logger.debug(f"Setting {section}.{option} to {value}")
        self.save()

    def save(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
        self.logger.debug(f"Configuration saved to {self.config_file}")

    def load_defaults(self, defaults):
        for section, options in defaults.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for option, value in options.items():
                if not self.config.has_option(section, option):
                    self.config.set(section, option, value)
                    self.logger.debug(f"Loading default {section}.{option} = {value}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    defaults = {
        'API': {
            'api_key': 'default_api_key',
            'base_url': 'https://api.example.com'
        },
        'DATABASE': {
            'db_name': 'default_db',
            'db_user': 'default_user'
        }
    }
    
    config_manager = ConfigManager(config_file='config.ini', defaults=defaults)
    config_manager.load_defaults(defaults)
    api_key = config_manager.get('API', 'api_key')
    base_url = config_manager.get('API', 'base_url')
    config_manager.set('API', 'timeout', '30')
