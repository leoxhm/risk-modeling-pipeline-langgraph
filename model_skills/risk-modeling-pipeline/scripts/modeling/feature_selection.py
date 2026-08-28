"""Train-only, auditable feature selection for LightGBM."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import polars as pl

from data.contract import ValidatedDataContract
from eda.analytics import EdaAnalysisResult, build_eda_analysis
from logger import get_logger

from .config import FeatureSelectionConfig


logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Selected features plus the evidence behind every decision."""

    feature_cols: tuple[str, ...]
    decisions: pl.DataFrame
    univariate_overview: pl.DataFrame
    binning_detail: pl.DataFrame


def _dominant_rate(series: pl.Series) -> float:
    values = series.drop_nulls().to_list()
    if not values:
        return 1.0
    return Counter(str(value) for value in values).most_common(1)[0][1] / len(
        values
    )


def _stability_by_feature(
    analysis: EdaAnalysisResult,
    config: FeatureSelectionConfig,
) -> dict[str, tuple[float, float, int]]:
    eligible_months = set(
        analysis.monthly_sample.filter(
            pl.col("sample_count") >= config.min_month_samples
        )
        .get_column("event_month")
        .cast(pl.String)
        .to_list()
    )
    result: dict[str, tuple[float, float, int]] = {}
    for feature in analysis.univariate_overview.get_column("feature").to_list():
        rows = analysis.monthly_psi.filter(
            (pl.col("feature") == feature)
            & pl.col("event_month").cast(pl.String).is_in(eligible_months)
            & (pl.col("event_month") != pl.col("baseline_month"))
        )
        psi_values = [float(value) for value in rows.get_column("psi").to_list()]
        max_psi = max(psi_values, default=0.0)
        unstable_count = sum(value > config.max_psi for value in psi_values)
        unstable_ratio = unstable_count / len(psi_values) if psi_values else 0.0
        result[feature] = (max_psi, unstable_ratio, len(psi_values))
    return result


def select_features(
    train_data: pl.DataFrame,
    contract: ValidatedDataContract,
    config: FeatureSelectionConfig,
) -> FeatureSelectionResult:
    """Apply quality, IV, stability, and redundancy rules using Train only."""
    analysis = build_eda_analysis(
        train_data,
        contract,
        correlation_method=config.correlation_method,
    )
    overview = analysis.univariate_overview
    iv_by_feature = dict(
        zip(overview.get_column("feature"), overview.get_column("iv"))
    )
    missing_by_feature = dict(
        zip(overview.get_column("feature"), overview.get_column("missing_rate"))
    )
    stability = _stability_by_feature(analysis, config)
    dominant_by_feature = {
        feature: _dominant_rate(train_data.get_column(feature))
        for feature in contract.feature_cols
    }
    unique_by_feature = {
        feature: train_data.get_column(feature).drop_nulls().n_unique()
        for feature in contract.feature_cols
    }
    correlation_lookup: dict[frozenset[str], float] = {
        frozenset((row["feature_1"], row["feature_2"])): float(
            row["abs_correlation"]
        )
        for row in analysis.correlation_pairs.to_dicts()
    }

    base_decisions: dict[str, tuple[str, str]] = {}
    candidates: list[str] = []
    for feature in contract.feature_cols:
        missing_rate = float(missing_by_feature[feature])
        iv = float(iv_by_feature[feature])
        max_psi, unstable_ratio, _ = stability[feature]
        unstable = (
            max_psi > config.max_psi
            and unstable_ratio > config.max_unstable_month_ratio
        )
        if unique_by_feature[feature] <= 1:
            base_decisions[feature] = (
                "excluded_constant",
                "non-null unique count <= 1",
            )
        elif missing_rate > config.max_missing_rate:
            base_decisions[feature] = (
                "excluded_missing",
                f"missing_rate > {config.max_missing_rate:.0%}",
            )
        elif iv < config.min_iv:
            base_decisions[feature] = (
                "excluded_low_iv",
                f"iv < {config.min_iv:.4f}",
            )
        elif unstable and config.stability_action == "drop":
            base_decisions[feature] = (
                "excluded_unstable",
                "monthly PSI exceeds configured level in too many eligible months",
            )
        else:
            candidates.append(feature)

    def priority(feature: str) -> tuple[bool, float, float, str]:
        max_psi, unstable_ratio, _ = stability[feature]
        unstable = (
            max_psi > config.max_psi
            and unstable_ratio > config.max_unstable_month_ratio
        )
        return (
            unstable,
            -float(iv_by_feature[feature]),
            float(missing_by_feature[feature]),
            feature,
        )

    ordered_candidates = sorted(candidates, key=priority)
    selected: list[str] = []
    correlated_with: dict[str, str] = {}
    for feature in ordered_candidates:
        retained = next(
            (
                candidate
                for candidate in selected
                if correlation_lookup.get(
                    frozenset((feature, candidate)), 0.0
                )
                > config.max_correlation
            ),
            None,
        )
        if retained is None:
            selected.append(feature)
        else:
            correlated_with[feature] = retained

    decision_rows: list[dict[str, object]] = []
    for feature in contract.feature_cols:
        max_psi, unstable_ratio, eligible_month_count = stability[feature]
        dominant_rate = dominant_by_feature[feature]
        if feature in base_decisions:
            decision, reason = base_decisions[feature]
        elif feature in correlated_with:
            decision = "excluded_correlation"
            reason = (
                f"abs_{config.correlation_method}_correlation > "
                f"{config.max_correlation:.2f} with {correlated_with[feature]}"
            )
        else:
            review_reasons: list[str] = []
            if (
                max_psi > config.max_psi
                and unstable_ratio > config.max_unstable_month_ratio
            ):
                review_reasons.append("stability")
            if dominant_rate > config.max_dominant_rate_warning:
                review_reasons.append("dominant value")
            if review_reasons:
                decision = "retained_review"
                reason = "Retained but requires review: " + ", ".join(
                    review_reasons
                )
            else:
                decision = "retained"
                reason = (
                    "Passed Train-only quality, IV, stability, and redundancy rules"
                )
        decision_rows.append(
            {
                "feature": feature,
                "decision": decision,
                "reason": reason,
                "iv": float(iv_by_feature[feature]),
                "missing_rate": float(missing_by_feature[feature]),
                "dominant_rate": dominant_rate,
                "max_monthly_psi": max_psi,
                "unstable_month_ratio": unstable_ratio,
                "eligible_month_count": eligible_month_count,
                "correlated_with": correlated_with.get(feature),
                "correlation_method": config.correlation_method,
            }
        )
    if not selected:
        raise ValueError(
            "No features remain after feature selection; revise model_config.yaml thresholds"
        )
    decisions = pl.DataFrame(decision_rows).sort(
        ["decision", "iv"], descending=[False, True]
    )
    logger.info(
        "Feature selection completed: retained=%d, excluded=%d",
        len(selected),
        len(decision_rows) - len(selected),
    )
    return FeatureSelectionResult(
        tuple(selected), decisions, overview, analysis.binning_detail
    )
