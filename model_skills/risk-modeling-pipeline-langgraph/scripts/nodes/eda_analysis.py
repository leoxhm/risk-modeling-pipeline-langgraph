"""LangGraph wrapper around the independent EDA core."""

from __future__ import annotations

from typing import Any

from core.data_read.contract import DataContract
from core.eda_analysis.metrics import run_eda


def run(*, data_path: str, output_dir: str, contract: DataContract) -> dict[str, Any]:
    """调用独立 EDA 核心逻辑；保留为后续节点编排的稳定入口。"""
    return run_eda(data_path=data_path, output_dir=output_dir, contract=contract)
