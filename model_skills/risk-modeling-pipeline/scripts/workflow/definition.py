"""Canonical eight-stage workflow contract shared by CLI, Skill, and UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowNode:
    """A user-facing stage; implementation details remain internal."""

    id: str
    name: str
    description: str
    confirmation_gate: str


WORKFLOW_NODES: tuple[WorkflowNode, ...] = (
    WorkflowNode(
        "data-read",
        "读取数据与契约校验",
        "读取文件，检查字段、主键、日期和目标标签",
        "sample",
    ),
    WorkflowNode(
        "eda-analysis",
        "EDA分析",
        "分析字段质量、样本分布、分箱、IV/KS、PSI 和相关性",
        "eda",
    ),
    WorkflowNode(
        "sample-diagnosis",
        "样本诊断",
        "基于 EDA 证据确认样本倾斜、月份异常、重复样本和样本处理策略",
        "sample",
    ),
    WorkflowNode(
        "feature-processing",
        "特征预处理与筛选",
        "识别类型，处理字符变量并按质量、IV、相关性和稳定性筛选",
        "feature",
    ),
    WorkflowNode(
        "model-config",
        "生成模型配置",
        "生成并解释切分、LightGBM 和 Optuna 参数",
        "model-config",
    ),
    WorkflowNode(
        "training-tuning",
        "模型训练与调参",
        "训练基线、运行调参并比较 Test/OOT 表现",
        "training-result",
    ),
    WorkflowNode(
        "model-review",
        "模型审查",
        "审查过拟合、稳定性、泄漏和 OOT 风险",
        "review",
    ),
    WorkflowNode(
        "report-delivery",
        "报告与模型交付",
        "生成 EDA/建模报告、模型文件和可复现运行产物",
        "delivery",
    ),
)

MAIN_NODE_IDS = frozenset(node.id for node in WORKFLOW_NODES)

# Internal events are intentionally retained for audit/debugging.  Consumers
# that show a user-facing flow should group them by this mapping instead of
# exposing every implementation detail as a separate node.
INTERNAL_TO_MAIN_NODE: dict[str, str] = {
    "loader": "data-read",
    "eda-analysis": "eda-analysis",
    "data-read-validation": "data-read",
    "profiler": "eda-analysis",
    "schema-review": "eda-analysis",
    "preflight": "eda-analysis",
    "experiment-plan": "model-config",
    "approval": "sample-diagnosis",
    "eda-cleaning": "eda-analysis",
    "sample-treatment": "sample-diagnosis",
    "cleaning": "feature-processing",
    "eda": "eda-analysis",
    "feature-selection": "feature-processing",
    "tuning": "training-tuning",
    "training": "training-tuning",
    "review": "model-review",
    "eda-report": "eda-analysis",
    "model-report": "report-delivery",
}

def main_node_for(node_id: str) -> str:
    """Return the user-facing stage for an internal or main node ID."""

    if node_id in MAIN_NODE_IDS:
        return node_id
    try:
        return INTERNAL_TO_MAIN_NODE[node_id]
    except KeyError as exc:
        raise ValueError(f"Unknown modeling node: {node_id}") from exc
