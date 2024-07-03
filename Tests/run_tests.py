# C:\TheTradingRobotPlug\Tests\run_tests.py

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Mock the configuration module
mock_config = MagicMock()
mock_config.__getitem__.side_effect = lambda key: {'loading_path': 'mock_path'}.get(key, {})

modules = {
    'Utilities.config_handling.config': mock_config,
    'Utilities.config_handling': MagicMock(config=mock_config)
}

with patch.dict('sys.modules', modules):
    # Import the test modules
    import Utilities.test_data_store

# Run the tests
unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(Utilities.test_data_store))
