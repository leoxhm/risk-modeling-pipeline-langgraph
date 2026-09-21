import os
import subprocess
import sys
import unittest
from pathlib import Path


class ConfirmCommandTest(unittest.TestCase):
    def test_unknown_node_is_rejected_before_node_execution(self):
        package_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "workflow", "confirm", "not-a-node"],
            cwd=package_root,
            env={**os.environ, "PYTHONPATH": str(package_root)},
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not confirmable", result.stderr + result.stdout)

    def test_later_nodes_accept_result_confirmation(self):
        package_root = Path(__file__).resolve().parents[2]
        # Later nodes do not expose a YAML form, but their completed result is
        # confirmed through the same command entry point.
        result = subprocess.run(
            [sys.executable, "-m", "workflow", "confirm", "eda-analysis", "--help"],
            cwd=package_root,
            env={**os.environ, "PYTHONPATH": str(package_root)},
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("confirm-result", result.stdout)

    def test_sample_diagnosis_accepts_result_confirmation(self):
        package_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "workflow", "confirm", "sample-diagnosis", "--help"],
            cwd=package_root,
            env={**os.environ, "PYTHONPATH": str(package_root)},
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("confirm-result", result.stdout)


if __name__ == "__main__":
    unittest.main()
