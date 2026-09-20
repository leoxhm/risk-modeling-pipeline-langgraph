"""本地训练监控日志，用于实时面板和后续模型审查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import threading
import time
from typing import Any


def _jsonable(value: Any) -> Any:
    """把 numpy 标量、Path 和嵌套容器转换成可写入 JSON 的对象。"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(value)


class TrainingMonitor:
    """以追加写入 JSONL 的方式保存每轮调参和拟合曲线。"""

    schema_version = 1

    def __init__(self, *, project_root: str | Path, run_id: str, method: str, output_dir: str | Path | None = None) -> None:
        """创建一个 run 级监控目录；不会为每个 trial 创建单独文件。"""
        base = Path(output_dir).expanduser().resolve() if output_dir else Path(project_root).expanduser().resolve() / "outputs" / run_id / "modeling-monitor"
        self.output_dir = base
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.method = method
        self.events_path = self.output_dir / "iterations.jsonl"
        self.meta_path = self.output_dir / "run.json"
        self._lock = threading.Lock()
        # 即使第一轮尚未完成也创建空日志，前端可立即打开监控页并轮询。
        self.events_path.touch(exist_ok=True)
        if not self.meta_path.exists():
            metadata = {
                "schema_version": self.schema_version,
                "run_id": run_id,
                "method": method,
                "status": "running",
                "created_at": time.time(),
                "events": str(self.events_path),
                "event_format": "jsonl",
                "metric_axes": ["train", "validate", "oot"],
                "notes": "每行一个 trial/迭代；fit_history 可选，用于绘制单次拟合曲线。",
            }
            self.meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_iteration(
        self,
        *,
        iteration: int,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, Any],
        status: str = "complete",
        fit_history: Mapping[str, Any] | None = None,
        accepted: bool | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        """追加一轮记录并立即 flush，供前端轮询 JSONL 文件。"""
        event = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method": self.method,
            "iteration": int(iteration),
            "timestamp": time.time(),
            "status": status,
            "parameters": _jsonable(parameters),
            "metrics": _jsonable(metrics),
            "fit_history": _jsonable(fit_history or {}),
            "accepted": accepted,
            "reason": reason,
            "duration_ms": duration_ms,
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        return event

    def finish(self, *, status: str = "completed", best_iteration: int | None = None) -> None:
        """更新 run 元数据，不额外产生一份重复的汇总文件。"""
        if not self.meta_path.exists():
            return
        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        metadata.update({"status": status, "finished_at": time.time(), "best_iteration": best_iteration})
        self.meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_progress(self, *, iteration: int, status: str, message: str | None = None) -> None:
        """更新当前轮次，供独立监控页在 LLM 请求期间显示实时进度。"""
        if not self.meta_path.exists():
            return
        with self._lock:
            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            metadata.update({"status": status, "current_iteration": int(iteration), "progress_message": message, "updated_at": time.time()})
            self.meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
