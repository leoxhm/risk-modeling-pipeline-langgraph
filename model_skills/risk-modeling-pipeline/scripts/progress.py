"""Structured node progress events for the modeling workflow."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import IO, Any, Iterator, Mapping, Sequence
from uuid import uuid4

from workflow import MAIN_NODE_IDS, main_node_for


INTERNAL_NODE_IDS = {
    "loader",
    "eda-analysis",
    "data-read-validation",
    "profiler",
    "schema-review",
    "preflight",
    "experiment-plan",
    "approval",
    "eda-cleaning",
    "sample-treatment",
    "cleaning",
    "eda",
    "feature-selection",
    "tuning",
    # Internal post-Optuna LLM refinement; mapped to the user-facing
    # model-config/training stage by workflow.definition.
    "llm-tuning",
    "training",
    "review",
    "eda-report",
    "model-report",
}
NODE_IDS = INTERNAL_NODE_IDS | set(MAIN_NODE_IDS)
NODE_STATUSES = {
    "pending",
    "running",
    "waiting_confirmation",
    "success",
    "failed",
    "skipped",
}
EVENT_PREFIX = "<!-- opencode-modeling-progress:"
EVENT_SUFFIX = " -->"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressReporter:
    """Persist an append-only event stream and a reconnectable state snapshot."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str | None = None,
        stream: IO[str] | None = sys.stdout,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "run_events.jsonl"
        self.state_path = self.output_dir / "run_state.json"
        self.stream = stream
        self.run_id = run_id or self.output_dir.name
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                value = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and isinstance(value.get("nodes"), dict):
                    value["run_id"] = self.run_id
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        timestamp = _now()
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "nodes": {},
        }

    def emit(
        self,
        node_id: str,
        status: str,
        *,
        summary: str | None = None,
        artifacts: Sequence[str | Path] = (),
        duration_ms: int | None = None,
        experiment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if node_id not in NODE_IDS:
            raise ValueError(f"Unknown modeling node: {node_id}")
        if status not in NODE_STATUSES:
            raise ValueError(f"Unknown modeling node status: {status}")
        timestamp = _now()
        event = {
            "schema_version": 1,
            "event_id": str(uuid4()),
            "run_id": self.run_id,
            "node_id": node_id,
            "main_node_id": main_node_for(node_id),
            "status": status,
            "timestamp": timestamp,
            "summary": summary,
            "artifacts": [str(Path(path).expanduser().resolve()) for path in artifacts],
            "duration_ms": duration_ms,
        }
        if experiment is not None:
            event["experiment"] = dict(experiment)
        with self.events_path.open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._state["updated_at"] = timestamp
        node_state = {
            "status": status,
            "main_node_id": event["main_node_id"],
            "summary": summary,
            "updated_at": timestamp,
            "artifacts": event["artifacts"],
            "duration_ms": duration_ms,
            "event_id": event["event_id"],
        }
        if experiment is not None:
            node_state["experiment"] = dict(experiment)
        elif node_id in self._state["nodes"] and "experiment" in self._state["nodes"][node_id]:
            node_state["experiment"] = self._state["nodes"][node_id]["experiment"]
        self._state["nodes"][node_id] = node_state
        temporary_path = self.state_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(self.state_path)
        if self.stream is not None:
            print(
                f"{EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}{EVENT_SUFFIX}",
                file=self.stream,
                flush=True,
            )
        return event

    @contextmanager
    def track(
        self,
        node_id: str,
        *,
        running_summary: str | None = None,
        success_summary: str | None = None,
        artifacts: Sequence[str | Path] = (),
    ) -> Iterator[None]:
        started_at = perf_counter()
        self.emit(node_id, "running", summary=running_summary)
        try:
            yield
            missing_artifacts = [
                str(Path(path).expanduser().resolve())
                for path in artifacts
                if not Path(path).expanduser().is_file()
                and not Path(path).expanduser().is_dir()
            ]
            if missing_artifacts:
                raise FileNotFoundError(f"Node {node_id} did not create declared artifacts: {missing_artifacts}")
        except BaseException as exc:
            self.emit(
                node_id,
                "failed",
                summary=str(exc) or exc.__class__.__name__,
                duration_ms=round((perf_counter() - started_at) * 1000),
            )
            raise
        self.emit(
            node_id,
            "success",
            summary=success_summary,
            artifacts=artifacts,
            duration_ms=round((perf_counter() - started_at) * 1000),
        )
