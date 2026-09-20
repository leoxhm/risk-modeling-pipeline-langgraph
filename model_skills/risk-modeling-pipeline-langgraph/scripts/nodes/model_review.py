"""Compatibility wrapper; model-review is wired in ``graph.stages``."""

from __future__ import annotations

from typing import Any


def run(state: dict[str, Any]) -> dict[str, Any]:
    """提示调用方使用带上下文的 Graph stage。"""
    raise NotImplementedError("Use WorkflowStages.prepare_model_review")
