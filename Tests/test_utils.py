import unittest
from unittest.mock import patch, Mock

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_requests_get = patch('requests.get').start()
        self.addCleanup(patch.stopall)
