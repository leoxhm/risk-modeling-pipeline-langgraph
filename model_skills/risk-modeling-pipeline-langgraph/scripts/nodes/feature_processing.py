"""LangGraph wrapper for the future feature-processing core implementation."""

from __future__ import annotations

from typing import Any


def run(state: dict[str, Any]) -> dict[str, Any]:
    """特征处理节点占位入口；当前版本不修改输入数据。"""
    raise NotImplementedError("Feature-processing core is the next implementation stage")
