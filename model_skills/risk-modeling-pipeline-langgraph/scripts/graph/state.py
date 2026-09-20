"""Serializable state carried between LangGraph nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class ModelingState(TypedDict, total=False):
    """Keep paths and decisions in state; never store a full DataFrame here."""

    run_id: str
    project_root: str
    skill_root: str
    selected_nodes: list[str]
    current_node: str
    status: str
    data_path: str | None
    data_read_result: dict[str, Any]
    # 当前运行的内部 IPC 缓存路径；只在 cache_data=true 时写入检查点。
    # 不把完整 DataFrame 写进 checkpoint，避免每个版本重复膨胀。
    data_cache: str | None
    cache_data: bool
    # 模型阶段只允许二选一：bayesian 或 llm。
    tuning_method: str | None
    model_config: dict[str, Any]
    optimizer_result: dict[str, Any]
    monitor_dir: str | None
    iteration_log: str | None
    pending: dict[str, Any]
    decision: dict[str, Any] | bool | None
    result: dict[str, Any]
    artifacts: dict[str, list[str]]
    summaries: dict[str, str]
    error: str | None
