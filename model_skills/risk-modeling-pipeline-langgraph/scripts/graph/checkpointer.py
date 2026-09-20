"""A small file-backed checkpointer for local OpenCode runs.

The official SQLite checkpointer is an optional package.  This skill keeps the
trial self-contained by persisting the in-memory saver dictionaries with
pickle.  It is intended for local, single-process-at-a-time use only.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
import threading
import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class PersistentMemorySaver(InMemorySaver):
    """Persist an ``InMemorySaver`` to a local file between ``start``/``resume``."""

    def __init__(self, path: str | Path) -> None:
        """初始化文件路径，并恢复已有的 LangGraph 检查点字典。"""
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        self._persist_lock = threading.Lock()
        if self.path.exists():
            with self.path.open("rb") as handle:
                payload = pickle.load(handle)
            self.storage.update(payload.get("storage", {}))
            self.writes.update(payload.get("writes", {}))
            self.blobs.update(payload.get("blobs", {}))

    def _persist(self) -> None:
        """以原子替换方式把检查点写入磁盘，避免进程中断产生半文件。"""
        with self._persist_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Deep-copy nested dicts to prevent "dictionary changed size
            # during iteration" when LangGraph mutates writes concurrently.
            payload = {
                "storage": copy.deepcopy(dict(self.storage)),
                "writes": copy.deepcopy(dict(self.writes)),
                "blobs": copy.deepcopy(dict(self.blobs)),
            }
            # LangGraph may commit checkpoints from worker threads.  A unique
            # temporary name prevents concurrent commits from replacing each
            # other's temporary file.
            tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            with tmp.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self.path)

    def put(self, *args: Any, **kwargs: Any):
        """保存单个检查点后立即持久化。"""
        result = super().put(*args, **kwargs)
        self._persist()
        return result

    def put_writes(self, *args: Any, **kwargs: Any) -> None:
        """保存待写入通道并同步持久化。"""
        super().put_writes(*args, **kwargs)
        self._persist()

    def delete_thread(self, thread_id: str) -> None:
        """删除指定线程的检查点，并更新持久化文件。"""
        super().delete_thread(thread_id)
        self._persist()
