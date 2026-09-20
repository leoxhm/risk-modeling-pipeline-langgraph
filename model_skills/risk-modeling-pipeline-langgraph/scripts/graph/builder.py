"""Declarative graph wiring for the LangGraph risk-modeling Skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .stages import WorkflowContext, WorkflowStages
from .state import ModelingState


NODE_ORDER = (
    "data-read",
    "eda-analysis",
    "sample-split",
    "feature-screening",
    "model-config",
    "bayesian-optimization",
    "llm-optimization",
    "model-review",
)


def _add_data_read_stage(builder: StateGraph, stages: WorkflowStages) -> None:
    """把 data-read 的准备、暂停和应用节点接入图。"""
    builder.add_node("data_read_prepare", stages.prepare_data_read)
    builder.add_node("data_read_wait", stages.wait_data_read)
    builder.add_node("data_read_apply", stages.apply_data_read)
    builder.add_edge(START, "data_read_prepare")
    builder.add_edge("data_read_prepare", "data_read_wait")
    builder.add_edge("data_read_wait", "data_read_apply")


def _add_eda_stage(builder: StateGraph, stages: WorkflowStages) -> None:
    """把 EDA/样本诊断的执行、暂停和应用节点接入图。"""
    builder.add_node("eda_run", stages.run_eda)
    builder.add_node("eda_wait", stages.wait_eda)
    builder.add_node("eda_apply", stages.apply_eda)
    builder.add_edge("data_read_apply", "eda_run")
    builder.add_edge("eda_run", "eda_wait")
    builder.add_edge("eda_wait", "eda_apply")
    builder.add_edge("eda_apply", "sample_split_prepare")


def _add_sample_split_stage(builder: StateGraph, stages: WorkflowStages) -> None:
    """把样本切分和特征筛选接入一次合并确认。"""
    builder.add_node("sample_split_prepare", stages.prepare_sample_split)
    builder.add_edge("sample_split_prepare", "feature_screening_prepare")


def _add_feature_screening_stage(builder: StateGraph, stages: WorkflowStages) -> None:
    """把特征筛选接入建模数据合并确认。"""
    builder.add_node("feature_screening_prepare", stages.prepare_feature_screening)
    builder.add_node("modeling_data_wait", stages.wait_modeling_data)
    builder.add_node("modeling_data_apply", stages.apply_modeling_data)
    builder.add_node("modeling_enter", stages.enter_modeling)
    builder.add_edge("feature_screening_prepare", "modeling_data_wait")
    builder.add_edge("modeling_data_wait", "modeling_data_apply")
    builder.add_edge("modeling_data_apply", "modeling_enter")
    builder.add_edge("modeling_enter", "model_config_prepare")


def _add_modeling_stage(builder: StateGraph, stages: WorkflowStages) -> None:
    """接入模型配置和二选一的调参分支，两个优化器不会并行执行。"""
    builder.add_node("model_config_prepare", stages.prepare_model_config)
    builder.add_node("model_config_wait", stages.wait_model_config)
    builder.add_node("model_config_apply", stages.apply_model_config)
    builder.add_node("bayesian_optimization", stages.run_bayesian_optimization)
    builder.add_node("llm_optimization", stages.run_llm_optimization)
    builder.add_node("model_review", stages.prepare_model_review)
    builder.add_edge("model_config_prepare", "model_config_wait")
    builder.add_edge("model_config_wait", "model_config_apply")
    builder.add_conditional_edges(
        "model_config_apply",
        stages.route_tuning_method,
        {
            "bayesian": "bayesian_optimization",
            "llm": "llm_optimization",
            "retry": "model_config_wait",
            "end": END,
        },
    )
    builder.add_edge("bayesian_optimization", "model_review")
    builder.add_edge("llm_optimization", "model_review")
    builder.add_edge("model_review", END)


def build_graph(*, project_root: str, skill_root: str, data: str | None, selected_nodes: list[str] | None = None, checkpointer: Any):
    """构建数据准备和模型调参图，调参阶段通过条件边严格二选一。"""
    # 当前版本 EDA 是 data-read 之后的必经阶段；保留 selected_nodes
    # 参数仅为兼容未来扩展，不允许跳过 EDA。
    _ = selected_nodes
    context = WorkflowContext(
        project_root=Path(project_root).expanduser().resolve(),
        skill_root=Path(skill_root).expanduser().resolve(),
        data=data,
    )
    stages = WorkflowStages(context)
    builder = StateGraph(ModelingState)
    _add_data_read_stage(builder, stages)
    _add_eda_stage(builder, stages)
    _add_sample_split_stage(builder, stages)
    _add_feature_screening_stage(builder, stages)
    _add_modeling_stage(builder, stages)
    return builder.compile(checkpointer=checkpointer)
