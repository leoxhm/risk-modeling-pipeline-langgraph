"""Tests for structured modeling progress events."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from progress import ProgressReporter


class ProgressReporterTest(unittest.TestCase):
    def test_writes_event_stream_snapshot_and_stdout_marker(self) -> None:
        stream = StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            reporter = ProgressReporter(temporary_directory, run_id="run-test", stream=stream)
            artifact = Path(temporary_directory) / "column_profile.csv"
            artifact.write_text("column\ncustomer_id\n", encoding="utf-8")
            reporter.emit(
                "profiler",
                "success",
                summary="完成字段画像",
                artifacts=[artifact],
                duration_ms=125,
                experiment={
                    "kind": "optuna",
                    "trial": 1,
                    "total_trials": 10,
                    "parameters": {"learning_rate": 0.03},
                },
            )

            events = [
                json.loads(line)
                for line in reporter.events_path.read_text(encoding="utf-8").splitlines()
            ]
            state = json.loads(reporter.state_path.read_text(encoding="utf-8"))

        self.assertEqual(events[0]["run_id"], "run-test")
        self.assertEqual(events[0]["node_id"], "profiler")
        self.assertEqual(events[0]["main_node_id"], "eda-analysis")
        self.assertEqual(events[0]["duration_ms"], 125)
        self.assertEqual(state["nodes"]["profiler"]["status"], "success")
        marker_text = stream.getvalue().removeprefix("<!-- opencode-modeling-progress:").removesuffix(" -->\n")
        marker = json.loads(marker_text)
        self.assertEqual(marker["event_id"], events[0]["event_id"])
        self.assertEqual(marker["run_id"], "run-test")
        self.assertEqual(marker["main_node_id"], "eda-analysis")
        self.assertEqual(marker["artifacts"], [str(artifact.resolve())])
        self.assertEqual(marker["duration_ms"], 125)
        self.assertEqual(marker["experiment"]["trial"], 1)
        self.assertEqual(state["nodes"]["profiler"]["experiment"]["kind"], "optuna")

    def test_track_records_failure_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reporter = ProgressReporter(temporary_directory, stream=None)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with reporter.track("training"):
                    raise RuntimeError("boom")
            state = json.loads(reporter.state_path.read_text(encoding="utf-8"))

        self.assertEqual(state["nodes"]["training"]["status"], "failed")
        self.assertEqual(state["nodes"]["training"]["summary"], "boom")
        self.assertIsInstance(state["nodes"]["training"]["duration_ms"], int)

    def test_restores_existing_state_for_a_follow_up_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ProgressReporter(temporary_directory, run_id="prepare-run", stream=None).emit(
                "approval", "waiting_confirmation"
            )
            reporter = ProgressReporter(temporary_directory, run_id="prepare-run", stream=None)
            reporter.emit("approval", "success", summary="用户确认完成")
            state = json.loads(reporter.state_path.read_text(encoding="utf-8"))

        self.assertEqual(state["nodes"]["approval"]["status"], "success")

    def test_does_not_mark_success_when_a_declared_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reporter = ProgressReporter(temporary_directory, stream=None)
            with self.assertRaises(FileNotFoundError):
                with reporter.track(
                    "model-report",
                    artifacts=[Path(temporary_directory) / "model_report.xlsx"],
                ):
                    pass
            state = json.loads(reporter.state_path.read_text(encoding="utf-8"))

        self.assertEqual(state["nodes"]["model-report"]["status"], "failed")
        self.assertIsInstance(state["nodes"]["model-report"]["duration_ms"], int)


if __name__ == "__main__":
    unittest.main()
