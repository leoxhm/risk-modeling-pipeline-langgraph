"""Conservative field-role and leakage review packet for a host LLM."""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from data.contract import ValidatedDataContract

from .schemas import Finding, finding_dicts, report_status


_OPAQUE_NAME = re.compile(r"^(flag|var|feature|x)_?\d+$", re.IGNORECASE)
_POST_OUTCOME_HINT = re.compile(
    r"(collection_result|final_status|write_?off|recovery_result|post_|after_|催收结果|最终状态|核销结果)",
    re.IGNORECASE,
)


def build_schema_proposal(
    column_profile: pl.DataFrame,
    contract: ValidatedDataContract,
) -> dict[str, Any]:
    """Return proposals and review questions; never silently change the contract."""

    raw = contract.contract
    rows = column_profile.to_dicts()
    findings: list[Finding] = []
    proposals: list[dict[str, Any]] = []
    feature_set = set(contract.feature_cols)

    opaque_features = [name for name in contract.feature_cols if _OPAQUE_NAME.match(name)]
    if opaque_features:
        findings.append(
            Finding(
                "FEATURE_SEMANTICS_MISSING",
                "warning",
                "多个候选特征名称缺少业务语义，AI 无法可靠判断时点泄漏。",
                {"columns": opaque_features},
                "补充字段中文含义、产生时点、来源系统和可用时点。",
            )
        )

    for row in rows:
        name = str(row["column_name"])
        confirmed_role = (
            "id"
            if name in raw.id_cols
            else "date"
            if name == raw.date_col
            else "target"
            if name == raw.target_col
            else "excluded"
            if name in raw.exclude_cols
            else "feature"
        )
        proposal = {
            "column_name": name,
            "confirmed_role": confirmed_role,
            "profile_role_suggestion": row.get("candidate_role"),
            "data_type": row.get("data_type"),
            "missing_rate": row.get("missing_rate"),
            "unique_rate": row.get("unique_rate"),
            "requires_semantic_review": False,
            "review_reason": None,
        }
        if name in feature_set and float(row.get("unique_rate") or 0.0) >= 0.98:
            proposal["requires_semantic_review"] = True
            proposal["review_reason"] = "高唯一率特征，疑似未声明的业务主键"
        if name in feature_set and _POST_OUTCOME_HINT.search(name):
            proposal["requires_semantic_review"] = True
            proposal["review_reason"] = "字段名包含贷后或结果语义，疑似时点泄漏"
        proposals.append(proposal)

    semantic_reviews = [row for row in proposals if row["requires_semantic_review"]]
    if semantic_reviews:
        findings.append(
            Finding(
                "SEMANTIC_FIELD_REVIEW_REQUIRED",
                "blocker",
                "存在需要人工确认业务含义的高风险字段。",
                {"columns": [row["column_name"] for row in semantic_reviews]},
                "确认字段产生时点并更新 schema.exclude_cols。",
            )
        )

    return {
        "status": report_status(findings),
        "proposals": proposals,
        "findings": finding_dicts(findings),
        "questions_for_user": [
            "这些字段在观察日前是否已经可用？",
            "是否存在贷后、催收、核销、最终审批结果字段？",
            "高唯一率字段是业务特征还是未声明主键？",
        ],
        "requires_user_confirmation": True,
    }
