import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import build_bootstrap_command


class AppHelperTests(unittest.TestCase):
    def test_bootstrap_command_points_to_demo_script(self):
        command = build_bootstrap_command()
        self.assertIn("bootstrap_demo_model.py", command)
        self.assertIn("python", command.lower())


if __name__ == "__main__":
    unittest.main()
