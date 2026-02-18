#!/usr/bin/env python3
"""Basic smoke tests for core module imports."""

import importlib
import unittest


class TestSmoke(unittest.TestCase):
    """Validate critical modules import without immediate runtime errors."""

    def test_import_core_modules(self):
        modules = [
            "config",
            "main",
            "run",
            "setup_wizard",
            "web_server",
            "modules.listener",
            "modules.brain",
            "modules.voice",
            "modules.face",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
