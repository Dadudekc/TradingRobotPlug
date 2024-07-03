# C:\TheTradingRobotPlug\Scripts\Utilities\config_handling.py

import configparser
import os

def load_config(config_file='config.ini'):
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

def load_paths(config):
    """
    Loads paths from the configuration file.

    Args:
        config (configparser.ConfigParser): ConfigParser object containing the configuration.

    Returns:
        dict: Dictionary with paths.
    """
    paths = {
        'data_folder': config.get('Paths', 'data_folder', fallback='/default/data/path'),
        'loading_path': config.get('Paths', 'loading_path', fallback='/default/loading/path'),
        'saving_path': config.get('Paths', 'saving_path', fallback='/default/saving/path')
    }
    return paths

def load_user_settings(config):
    """
    Loads user settings from the configuration file.

    Args:
        config (configparser.ConfigParser): ConfigParser object containing the configuration.

    Returns:
        dict: Dictionary with user settings.
    """
    if 'UserSettings' in config:
        return dict(config['UserSettings'])
    else:
        return {}
