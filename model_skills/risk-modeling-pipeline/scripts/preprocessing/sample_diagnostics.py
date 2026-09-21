"""Deterministic raw-sample diagnostics and confirmed treatment actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import median
from typing import Any

import polars as pl

from data.contract import DataContract

from .sample_config import SampleConfig


_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m", "%Y-%m")


@dataclass(frozen=True)
class SampleFinding:
    """One deterministic sample finding consumed by the AI advisory layer."""

    code: str
    severity: str
    message: str
    evidence: dict[str, Any]
    recommendation: str | None


def _report_status(findings: list[SampleFinding]) -> str:
    """Return an advisory status without gating later workflow stages.

    Sample diagnosis is an evidence/confirmation step.  Findings are useful
    warnings, but they must not turn into a workflow blocker: the user may
    choose to continue and let the later preprocessing/model configuration
    apply the confirmed policy.  Keep the old ``blocker`` severity readable
    for backward-compatible artifacts, while exposing a non-blocking status.
    """
    return "needs_review" if findings else "ready"


@dataclass(frozen=True)
class SampleTreatmentResult:
    """Treated data and an auditable count of every sample-level change."""

    data: pl.DataFrame
    removed_duplicate_count: int
    removed_missing_target_count: int
    removed_high_missing_row_count: int
    removed_incomplete_month_count: int
    removed_all_null_features: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "output_row_count": self.data.height,
            "removed_duplicate_count": self.removed_duplicate_count,
            "removed_missing_target_count": self.removed_missing_target_count,
            "removed_high_missing_row_count": self.removed_high_missing_row_count,
            "removed_incomplete_month_count": self.removed_incomplete_month_count,
            "removed_all_null_features": list(self.removed_all_null_features),
        }


def _date_expression(column: str) -> pl.Expr:
    normalized = pl.col(column).cast(pl.String).str.strip_chars()
    return pl.coalesce(
        [
            normalized.str.strptime(pl.Date, format=date_format, strict=False)
            for date_format in _DATE_FORMATS
        ]
    )


def _candidate_features(data: pl.DataFrame, contract: DataContract) -> list[str]:
    roles = (
        set(contract.id_cols)
        | {contract.date_col, contract.target_col}
        | set(contract.exclude_cols)
    )
    return [column for column in data.columns if column not in roles]


def _duplicate_metrics(data: pl.DataFrame, id_cols: tuple[str, ...]) -> dict[str, int]:
    groups = data.group_by(list(id_cols)).len().filter(pl.col("len") > 1)
    duplicate_group_count = groups.height
    duplicate_row_count = (
        int(groups.get_column("len").sum() or 0) if duplicate_group_count else 0
    )
    return {
        "duplicate_key_count": duplicate_group_count,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_excess_row_count": duplicate_row_count - duplicate_group_count,
    }


def _month_sequence(start: str, end: str) -> list[str]:
    start_date = datetime.strptime(start, "%Y%m")
    end_date = datetime.strptime(end, "%Y%m")
    values: list[str] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        values.append(f"{year:04d}{month:02d}")
        month = month + 1
        if month == 13:
            year, month = year + 1, 1
    return values


def monthly_target_summary(data: pl.DataFrame, contract: DataContract) -> pl.DataFrame:
    """Return raw and labeled sample counts by parsable observation month."""

    return (
        data.with_columns(
            _date_expression(contract.date_col)
            .dt.strftime("%Y%m")
            .alias("event_month"),
            (pl.col(contract.target_col) == pl.lit(contract.good_label))
            .fill_null(False)
            .cast(pl.Int64)
            .alias("__is_good"),
            (pl.col(contract.target_col) == pl.lit(contract.bad_label))
            .fill_null(False)
            .cast(pl.Int64)
            .alias("__is_bad"),
            pl.col(contract.target_col)
            .is_null()
            .cast(pl.Int64)
            .alias("__target_missing"),
            (
                pl.col(contract.target_col).is_not_null()
                & ~pl.col(contract.target_col).is_in(
                    [contract.good_label, contract.bad_label]
                )
            )
            .cast(pl.Int64)
            .alias("__target_invalid"),
        )
        .filter(pl.col("event_month").is_not_null())
        .group_by("event_month")
        .agg(
            pl.len().alias("sample_count"),
            pl.col("__is_good").sum().alias("good_count"),
            pl.col("__is_bad").sum().alias("bad_count"),
            pl.col("__target_missing").sum().alias("missing_target_count"),
            pl.col("__target_invalid").sum().alias("invalid_target_count"),
        )
        .with_columns(
            (pl.col("good_count") + pl.col("bad_count")).alias("labeled_sample_count"),
        )
        .with_columns(
            pl.when(pl.col("labeled_sample_count") > 0)
            .then(pl.col("bad_count") / pl.col("labeled_sample_count"))
            .otherwise(None)
            .alias("bad_rate")
        )
        .sort("event_month")
    )


def _latest_month_is_incomplete(
    monthly: pl.DataFrame, minimum_ratio: float
) -> tuple[bool, dict[str, Any]]:
    if monthly.height < 2:
        return False, {}
    rows = monthly.to_dicts()
    latest = rows[-1]
    latest_count = int(latest["sample_count"])
    history = [int(row["sample_count"]) for row in rows[:-1]]
    reference = float(median(history[-3:]))
    evidence = {
        "latest_month": latest["event_month"],
        "latest_sample_count": latest_count,
        "recent_median_sample_count": reference,
        "minimum_volume_ratio": minimum_ratio,
    }
    return reference > 0 and latest_count < minimum_ratio * reference, evidence


def build_sample_diagnostics(
    data: pl.DataFrame,
    contract: DataContract,
    config: SampleConfig,
) -> dict[str, Any]:
    """Calculate reviewable sample evidence without changing the source data."""

    features = _candidate_features(data, contract)
    treatment = config.treatment
    thresholds = config.diagnostics
    duplicate_metrics = _duplicate_metrics(data, contract.id_cols)
    target = data.get_column(contract.target_col)
    missing_target_count = target.null_count()
    invalid_target_count = int(
        (
            target.is_not_null()
            & ~target.is_in([contract.good_label, contract.bad_label])
        ).sum()
    )
    good_count = int((target == contract.good_label).fill_null(False).sum())
    bad_count = int((target == contract.bad_label).fill_null(False).sum())
    labeled_count = good_count + bad_count
    minority_count = min(good_count, bad_count)
    minority_rate = minority_count / labeled_count if labeled_count else 0.0
    imbalance_ratio = (
        max(good_count, bad_count) / minority_count if minority_count else None
    )
    all_null_features = [
        column
        for column in features
        if data.get_column(column).null_count() == data.height
    ]

    if features:
        missing_count = pl.sum_horizontal(
            [pl.col(column).is_null().cast(pl.Int64) for column in features]
        )
        high_missing_row_count = int(
            data.select(
                (
                    missing_count / len(features) >= thresholds.high_missing_row_rate
                ).sum()
            ).item()
        )
    else:
        high_missing_row_count = 0

    monthly = monthly_target_summary(data, contract)
    month_values = monthly.get_column("event_month").to_list() if monthly.height else []
    missing_months = (
        sorted(
            set(_month_sequence(month_values[0], month_values[-1])) - set(month_values)
        )
        if len(month_values) >= 2
        else []
    )
    monthly_rows = monthly.to_dicts()
    rate_changes = []
    for previous, current in zip(monthly_rows, monthly_rows[1:]):
        if previous["bad_rate"] is None or current["bad_rate"] is None:
            continue
        rate_changes.append(
            {
                "previous_month": previous["event_month"],
                "current_month": current["event_month"],
                "previous_bad_rate": previous["bad_rate"],
                "current_bad_rate": current["bad_rate"],
                "absolute_change": abs(current["bad_rate"] - previous["bad_rate"]),
            }
        )
    largest_rate_change = max(
        rate_changes, key=lambda item: item["absolute_change"], default=None
    )
    latest_incomplete, latest_evidence = _latest_month_is_incomplete(
        monthly, thresholds.latest_month_min_volume_ratio
    )

    findings: list[SampleFinding] = []
    if missing_target_count:
        findings.append(
            SampleFinding(
                "MISSING_TARGET",
                "warning",
                "存在缺失标签样本。",
                {
                    "count": missing_target_count,
                    "rate": missing_target_count / data.height,
                },
                "补充标签，或确认仅删除缺失标签的样本。",
            )
        )
    if invalid_target_count:
        findings.append(
            SampleFinding(
                "INVALID_TARGET_VALUES",
                "warning",
                "目标字段包含好坏标签映射之外的取值。",
                {"count": invalid_target_count},
                "修正标签口径或更新标签映射；本节点仅告警，不自动改写标签，也不阻止后续节点执行。",
            )
        )
    if duplicate_metrics["duplicate_key_count"]:
        findings.append(
            SampleFinding(
                "DUPLICATE_ID",
                "warning",
                "配置的主键存在重复记录。",
                duplicate_metrics,
                "确认重复记录的业务含义及保留第一条或最后一条，并在报告中记录删除数量。",
            )
        )
    if all_null_features:
        findings.append(
            SampleFinding(
                "ALL_NULL_FEATURES",
                "warning",
                "存在全空候选特征。",
                {"columns": all_null_features},
                "确认删除全空特征，并检查上游取数逻辑。",
            )
        )
    if high_missing_row_count:
        findings.append(
            SampleFinding(
                "HIGH_MISSING_ROWS",
                "warning",
                "部分样本在候选特征上的缺失比例过高。",
                {
                    "count": high_missing_row_count,
                    "rate": high_missing_row_count / data.height,
                    "threshold": thresholds.high_missing_row_rate,
                },
                "确认保留这些样本，或仅在用户批准后删除。",
            )
        )
    if labeled_count and minority_rate < thresholds.imbalance_warning_minority_rate:
        # Even an extremely small minority class is advisory at this stage.
        # The critical threshold remains in the evidence so the modeling
        # configuration can make an explicit, user-approved choice later.
        severity = "warning"
        findings.append(
            SampleFinding(
                "CLASS_IMBALANCE_CRITICAL"
                if minority_rate < thresholds.imbalance_critical_minority_rate
                else "CLASS_IMBALANCE_WARNING",
                severity,
                "正负样本分布不均衡。",
                {
                    "good_count": good_count,
                    "bad_count": bad_count,
                    "minority_rate": minority_rate,
                    "majority_to_minority_ratio": imbalance_ratio,
                    "critical_threshold": thresholds.imbalance_critical_minority_rate,
                    "critical_threshold_breached": minority_rate < thresholds.imbalance_critical_minority_rate,
                },
                "后续模型配置阶段确认是否仅在 Train 上使用类别权重；不要改变 Test/OOT 的自然分布。",
            )
        )
    if missing_months:
        findings.append(
            SampleFinding(
                "MISSING_MONTHS",
                "warning",
                "观察期内存在月份断层。",
                {"months": missing_months},
                "核对上游数据是否漏取；确认前不要自动补齐月份。",
            )
        )
    if (
        largest_rate_change
        and largest_rate_change["absolute_change"]
        >= thresholds.monthly_bad_rate_change_warning
    ):
        findings.append(
            SampleFinding(
                "MONTHLY_BAD_RATE_SHIFT",
                "warning",
                "相邻月份坏样本率波动超过阈值。",
                largest_rate_change,
                "检查标签成熟度、客群变化和策略口径变更。",
            )
        )
    if latest_incomplete:
        findings.append(
            SampleFinding(
                "LATEST_MONTH_INCOMPLETE",
                "warning",
                "最新月样本量明显低于近期水平，疑似数据未完整。",
                latest_evidence,
                "确认数据截止日期，或明确选择保留/排除最新月份。",
            )
        )

    metrics = {
        "row_count": data.height,
        "column_count": data.width,
        "candidate_feature_count": len(features),
        "good_count": good_count,
        "bad_count": bad_count,
        "labeled_sample_count": labeled_count,
        "missing_target_count": missing_target_count,
        "invalid_target_count": invalid_target_count,
        "bad_rate": bad_count / labeled_count if labeled_count else None,
        "minority_rate": minority_rate if labeled_count else None,
        "majority_to_minority_ratio": imbalance_ratio,
        **duplicate_metrics,
        "all_null_feature_count": len(all_null_features),
        "high_missing_row_count": high_missing_row_count,
        "missing_month_count": len(missing_months),
    }
    return {
        "status": _report_status(findings),
        "metrics": metrics,
        "monthly_target_summary": monthly_rows,
        "findings": [asdict(finding) for finding in findings],
        "proposed_treatment": asdict(treatment),
        "requires_user_confirmation": True,
    }


def apply_sample_treatment(
    data: pl.DataFrame,
    contract: DataContract,
    config: SampleConfig,
) -> SampleTreatmentResult:
    """Apply only the treatment actions already covered by an approval manifest."""

    result = data
    treatment = config.treatment
    features = _candidate_features(result, contract)

    missing_target_count = result.get_column(contract.target_col).null_count()
    if missing_target_count:
        result = result.filter(pl.col(contract.target_col).is_not_null())

    duplicate_metrics = _duplicate_metrics(result, contract.id_cols)
    duplicate_excess = duplicate_metrics["duplicate_excess_row_count"]
    if duplicate_excess:
        # ``error`` is accepted only as a legacy alias. New templates use an
        # explicit deterministic keep action so data issues do not abort an
        # interactive run.
        keep = "first" if treatment.duplicate_action != "keep_last" else "last"
        result = result.unique(
            subset=list(contract.id_cols), keep=keep, maintain_order=True
        )

    all_null_features = tuple(
        column
        for column in features
        if result.get_column(column).null_count() == result.height
    )
    if all_null_features:
        result = result.drop(list(all_null_features))
    retained_features = [
        column for column in features if column not in all_null_features
    ]

    removed_high_missing_row_count = 0
    if retained_features and treatment.high_missing_row_action == "drop":
        missing_count = pl.sum_horizontal(
            [pl.col(column).is_null().cast(pl.Int64) for column in retained_features]
        )
        keep_expression = (
            missing_count / len(retained_features)
            < config.diagnostics.high_missing_row_rate
        )
        original_count = result.height
        result = result.filter(keep_expression)
        removed_high_missing_row_count = original_count - result.height

    monthly = monthly_target_summary(result, contract)
    incomplete, evidence = _latest_month_is_incomplete(
        monthly, config.diagnostics.latest_month_min_volume_ratio
    )
    removed_incomplete_month_count = 0
    # ``error`` remains a legacy alias for the safe default ``exclude``.
    if incomplete and treatment.incomplete_latest_month_action != "keep":
        parsed_month = _date_expression(contract.date_col).dt.strftime("%Y%m")
        original_count = result.height
        result = result.filter(
            parsed_month.is_null() | (parsed_month != evidence["latest_month"])
        )
        removed_incomplete_month_count = original_count - result.height

    return SampleTreatmentResult(
        data=result,
        removed_duplicate_count=duplicate_excess,
        removed_missing_target_count=missing_target_count,
        removed_high_missing_row_count=removed_high_missing_row_count,
        removed_incomplete_month_count=removed_incomplete_month_count,
        removed_all_null_features=all_null_features,
    )
