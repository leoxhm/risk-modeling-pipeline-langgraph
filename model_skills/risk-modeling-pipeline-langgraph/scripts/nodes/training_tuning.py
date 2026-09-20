"""Compatibility wrapper; training branches are wired in ``graph.stages``."""

from __future__ import annotations

from typing import Any


def run(state: dict[str, Any]) -> dict[str, Any]:
    """提示调用方使用带上下文的 Graph stage，而不是丢失项目路径。"""
    raise NotImplementedError("Use WorkflowStages.run_bayesian_optimization or run_llm_optimization")
