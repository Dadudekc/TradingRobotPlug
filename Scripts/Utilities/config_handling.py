# C:\TheTradingRobotPlug\Scripts\Utilities\config_handling.py

import configparser
import os

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.config['DEFAULT'] = {}

    def get(self, section, option, fallback=None):
        return self.config.get(section, option, fallback=fallback)

    def set(self, section, option, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def save(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
