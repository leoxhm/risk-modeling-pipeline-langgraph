"""Deterministic evidence packet for an AI preflight modeling review."""

from __future__ import annotations

from typing import Any

import polars as pl

from data.contract import ValidatedDataContract
from modeling.config import ModelConfig

from .schemas import Finding, finding_dicts, report_status


def build_preflight_report(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    column_profile: pl.DataFrame,
    model_config: ModelConfig | None,
    sample_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build facts and guardrail findings without asking an LLM to calculate them."""

    monthly_rows = sample_diagnostics["monthly_target_summary"]
    raw = contract.contract
    bad_count = int((data.get_column(raw.target_col) == raw.bad_label).sum())
    findings = [Finding(**finding) for finding in sample_diagnostics["findings"]]

    if data.height == 0:
        findings.append(Finding("EMPTY_DATASET", "blocker", "数据集为空，无法建模。"))
    elif bad_count < 30:
        findings.append(
            Finding(
                "TOTAL_BAD_COUNT_CRITICAL",
                "blocker",
                "总坏样本数少于 30，无法支撑稳定的模型评估。",
                {"bad_count": bad_count},
                "补充或延长成熟标签窗口。",
            )
        )
    elif bad_count < 100:
        findings.append(
            Finding(
                "TOTAL_BAD_COUNT_LOW",
                "warning",
                "总坏样本数偏少，复杂模型和大规模调参容易不稳定。",
                {"bad_count": bad_count},
                "限制模型复杂度和调参空间。",
            )
        )

    if model_config is not None and len(monthly_rows) > model_config.split.oot_months:
        oot_rows = monthly_rows[-model_config.split.oot_months :]
        oot_bad_count = sum(int(row["bad_count"]) for row in oot_rows)
        oot_sample_count = sum(int(row["sample_count"]) for row in oot_rows)
        evidence = {
            "oot_months": [row["event_month"] for row in oot_rows],
            "oot_sample_count": oot_sample_count,
            "oot_bad_count": oot_bad_count,
        }
        if oot_bad_count < 10:
            findings.append(
                Finding(
                    "OOT_BAD_COUNT_CRITICAL",
                    "blocker",
                    "OOT 坏样本少于 10，AUC、KS 不具备稳定评价基础。",
                    evidence,
                    "重新选择标签已成熟的 OOT 时间窗口。",
                )
            )
        elif oot_bad_count < 30:
            findings.append(
                Finding(
                    "OOT_BAD_COUNT_LOW",
                    "warning",
                    "OOT 坏样本少于 30，指标仅建议作方向性参考。",
                    evidence,
                    "增加 OOT 坏样本后再作生产准入判断。",
                )
            )

    high_missing = (
        column_profile.filter(pl.col("missing_rate") > 0.8)
        .get_column("column_name")
        .to_list()
    )
    if high_missing:
        findings.append(
            Finding(
                "HIGH_MISSING_COLUMNS",
                "warning",
                "存在空值率高于 80% 的字段。",
                {"columns": high_missing},
                "在 Train 上拟合字段去留规则，再应用到 Test/OOT。",
            )
        )

    summary = {
        "row_count": data.height,
        "column_count": data.width,
        "candidate_feature_count": len(contract.feature_cols),
        "bad_count": bad_count,
        "bad_rate": bad_count / data.height if data.height else None,
        "id_unique_rate": contract.id_unique_rate,
        "date_parse_rate": contract.date_parse_rate,
    }
    return {
        "status": report_status(findings),
        "summary": summary,
        "monthly_target_summary": monthly_rows,
        "findings": finding_dicts(findings),
        "sample_diagnostics": sample_diagnostics,
        "requires_user_confirmation": True,
    }
