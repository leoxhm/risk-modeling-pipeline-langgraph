"""Deterministic EDA calculations for structured binary-classification data."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any, Iterable

import polars as pl

from data.contract import ValidatedDataContract
from logger import get_logger
from metrics import (
    resolve_backend,
    toad_iv_from_bins,
    toad_ks,
    toad_ks_bucket,
    toad_psi,
)


logger = get_logger(__name__)
_EPSILON = 0.5


@dataclass(frozen=True)
class EdaAnalysisResult:
    """All deterministic tables required by the data EDA workbook."""

    univariate_overview: pl.DataFrame
    binning_detail: pl.DataFrame
    monthly_sample: pl.DataFrame
    monthly_psi: pl.DataFrame
    correlation_pairs: pl.DataFrame
    monthly_discrimination: pl.DataFrame
    ks_bucket: pl.DataFrame


def _is_numeric(series: pl.Series) -> bool:
    return series.dtype.is_numeric() and series.dtype != pl.Boolean


def _quantile_edges(values: list[float], bin_count: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    edges = {
        ordered[min(len(ordered) - 1, round(index * (len(ordered) - 1) / bin_count))]
        for index in range(1, bin_count)
    }
    return sorted(edges)


def _numeric_binner(values: Iterable[Any], bin_count: int) -> tuple[dict[Any, int], list[str]]:
    numeric_values = [float(value) for value in values if value is not None]
    edges = _quantile_edges(numeric_values, bin_count)
    labels: list[str] = []
    for index, edge in enumerate(edges):
        left = "-inf" if index == 0 else f"{edges[index - 1]:.6g}"
        labels.append(f"({left}, {edge:.6g}]")
    labels.append(f"({edges[-1]:.6g}, inf)" if edges else "all")
    # Mapping raw values avoids recalculating boundaries for each use in a small EDA dataset.
    return {value: bisect_right(edges, float(value)) for value in set(numeric_values)}, labels


def _categorical_binner(values: Iterable[Any], max_categories: int) -> tuple[dict[Any, int], list[str]]:
    counts = Counter(str(value) for value in values if value is not None)
    categories = [name for name, _ in counts.most_common(max_categories - 1)]
    labels = categories + (["OTHER"] if len(counts) > len(categories) else [])
    mapping = {
        value: categories.index(str(value)) if str(value) in categories else len(categories)
        for value in set(values)
        if value is not None
    }
    return mapping, labels


def _rank_auc(values: list[float], targets: list[int]) -> float | None:
    bad_count = sum(targets)
    good_count = len(targets) - bad_count
    if not bad_count or not good_count:
        return None
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for original_index, _ in indexed[start:end]:
            ranks[original_index] = average_rank
        start = end
    bad_rank_sum = sum(rank for rank, target in zip(ranks, targets) if target == 1)
    raw_auc = (bad_rank_sum - bad_count * (bad_count + 1) / 2) / (bad_count * good_count)
    return max(raw_auc, 1 - raw_auc)


def _bin_feature(
    data: pl.DataFrame,
    feature: str,
    target_col: str,
    *,
    bin_count: int,
    max_categories: int,
    metrics_backend: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int | None], dict[Any, int], list[str]]:
    series = data.get_column(feature)
    targets = data.get_column(target_col).to_list()
    raw_values = series.to_list()
    numeric = _is_numeric(series)
    if numeric:
        mapping, labels = _numeric_binner(raw_values, bin_count)
    else:
        mapping, labels = _categorical_binner(raw_values, max_categories)

    bin_values = [-1 if value is None else mapping[value] for value in raw_values]
    toad_iv: float | None = None
    toad_iv_by_bin: dict[int, float] = {}
    toad_ks_value: float | None = None
    if metrics_backend == "toad":
        toad_iv, toad_iv_by_bin = toad_iv_from_bins(bin_values, targets)
        # Toad's KS is a two-sample statistic over a score-like array.  For a
        # numeric feature use the original (non-null, finite) values so the
        # headline KS is the standard feature KS, rather than a KS calculated
        # from arbitrary integer bin IDs.  Categorical values do not have a
        # natural score order, therefore retain the deterministic bin order as
        # a comparative diagnostic for those fields.
        if numeric:
            finite_pairs = [
                (float(value), int(target))
                for value, target in zip(raw_values, targets)
                if value is not None and math.isfinite(float(value))
            ]
            toad_ks_value = toad_ks(
                [value for value, _ in finite_pairs],
                [target for _, target in finite_pairs],
            )
        else:
            toad_ks_value = toad_ks(bin_values, targets)

    groups: dict[int, dict[str, int]] = defaultdict(lambda: {"sample_count": 0, "bad_count": 0})
    for value, target in zip(raw_values, targets):
        bin_id = -1 if value is None else mapping[value]
        groups[bin_id]["sample_count"] += 1
        groups[bin_id]["bad_count"] += int(target)

    # Missing values are retained as an explicit reporting bin, but are not
    # appended to the ordered distribution used by KS.  Appending them after
    # every regular bin makes both cumulative distributions end at 100%, which
    # would mechanically report KS=1 for almost every feature.
    ordered_regular_bin_ids = sorted(bin_id for bin_id in groups if bin_id >= 0)
    ordered_bin_ids = list(ordered_regular_bin_ids)
    if -1 in groups:
        ordered_bin_ids.append(-1)
    total_bad = sum(int(target) for target in targets)
    total_good = len(targets) - total_bad
    group_count = len(ordered_bin_ids)
    cumulative_bad = 0
    cumulative_good = 0
    iv_total = 0.0
    ks_max = 0.0
    details: list[dict[str, Any]] = []
    for order, bin_id in enumerate(ordered_bin_ids, start=1):
        group = groups[bin_id]
        sample_count = group["sample_count"]
        bad_count = group["bad_count"]
        good_count = sample_count - bad_count
        bad_dist = (bad_count + _EPSILON) / (total_bad + _EPSILON * group_count)
        good_dist = (good_count + _EPSILON) / (total_good + _EPSILON * group_count)
        woe = math.log(bad_dist / good_dist)
        legacy_iv_value = (bad_dist - good_dist) * woe
        iv_value = (
            toad_iv_by_bin.get(bin_id, legacy_iv_value)
            if metrics_backend == "toad"
            else legacy_iv_value
        )
        iv_total += iv_value
        if bin_id >= 0:
            cumulative_bad += bad_count
            cumulative_good += good_count
            ks_value: float | None = abs(
                (cumulative_bad / total_bad if total_bad else 0.0)
                - (cumulative_good / total_good if total_good else 0.0)
            )
            ks_max = max(ks_max, ks_value)
        else:
            ks_value = None
        label = "MISSING" if bin_id == -1 else labels[bin_id]
        details.append(
            {
                "feature": feature,
                "bin_order": order,
                "bin": label,
                "sample_count": sample_count,
                "sample_rate": sample_count / len(targets) if targets else 0.0,
                "bad_count": bad_count,
                "good_count": good_count,
                "bad_rate": bad_count / sample_count if sample_count else 0.0,
                "bad_distribution": bad_count / total_bad if total_bad else 0.0,
                "good_distribution": good_count / total_good if total_good else 0.0,
                "woe": woe,
                "iv_bin": iv_value,
                "ks": ks_value,
            }
        )

    numeric_values = (
        [float(value) for value in raw_values if value is not None]
        if numeric
        else []
    )
    numeric_targets = (
        [
            int(target)
            for value, target in zip(raw_values, targets)
            if value is not None
        ]
        if numeric
        else []
    )
    metrics: dict[str, float | int | None] = {
        "iv": toad_iv if metrics_backend == "toad" else iv_total,
        "ks": toad_ks_value if metrics_backend == "toad" else ks_max,
        "auc_adjusted": _rank_auc(numeric_values, numeric_targets) if numeric else None,
        "bin_count": len(ordered_bin_ids),
        "missing_rate": series.null_count() / data.height if data.height else 0.0,
    }
    return details, metrics, mapping, labels


def _feature_psi(
    data: pl.DataFrame,
    feature: str,
    month_col: str,
    mapping: dict[Any, int],
    labels: list[str],
    baseline_month: str,
    metrics_backend: str,
) -> list[dict[str, Any]]:
    values = data.get_column(feature).to_list()
    months = data.get_column(month_col).cast(pl.String).to_list()
    bins = [-1 if value is None else mapping.get(value, len(labels)) for value in values]
    month_bin_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for month, bin_id in zip(months, bins):
        if month is not None:
            month_bin_counts[month][bin_id] += 1
    baseline = month_bin_counts[baseline_month]
    all_bin_ids = set(bins)
    baseline_total = sum(baseline.values())
    rows: list[dict[str, Any]] = []
    for month in sorted(month_bin_counts):
        actual = month_bin_counts[month]
        if metrics_backend == "toad":
            # PSI must compare the rows belonging to the current month with
            # the rows belonging to the baseline month.  Passing the global
            # bin vector here makes every month compare the same distribution
            # and silently produces PSI=0.  Filter the fitted bins by month,
            # while keeping the complete support so missing buckets remain
            # auditable.
            actual_bins = [
                bin_id
                for month_value, bin_id in zip(months, bins)
                if month_value == month
            ]
            baseline_bins = [
                bin_id
                for month_value, bin_id in zip(months, bins)
                if month_value == baseline_month
            ]
            psi = toad_psi(
                actual_bins,
                baseline_bins,
                support=all_bin_ids,
            )
            psi = float(psi) if psi is not None else 0.0
        else:
            actual_total = sum(actual.values())
            psi = 0.0
            for bin_id in all_bin_ids:
                base_share = (baseline[bin_id] + _EPSILON) / (baseline_total + _EPSILON * len(all_bin_ids))
                actual_share = (actual[bin_id] + _EPSILON) / (actual_total + _EPSILON * len(all_bin_ids))
                psi += (actual_share - base_share) * math.log(actual_share / base_share)
        rows.append(
            {
                "feature": feature,
                "baseline_month": baseline_month,
                "event_month": month,
                "psi": psi,
                "metrics_backend": metrics_backend,
            }
        )
    return rows


def _metric_ks(scores: list[float], targets: list[int], metrics_backend: str) -> float | None:
    """Return KS using the configured backend, with a deterministic fallback."""
    if len(scores) != len(targets) or len(set(targets)) < 2:
        return None
    if metrics_backend == "toad":
        return toad_ks(scores, targets)
    ordered = sorted(zip(scores, targets), key=lambda item: item[0], reverse=True)
    total_bad = sum(targets)
    total_good = len(targets) - total_bad
    bad_seen = good_seen = 0
    maximum = 0.0
    for _, target in ordered:
        if target:
            bad_seen += 1
        else:
            good_seen += 1
        maximum = max(
            maximum,
            abs(bad_seen / total_bad - good_seen / total_good),
        )
    return maximum


def _monthly_discrimination(
    data: pl.DataFrame,
    feature: str,
    target_col: str,
    month_col: str,
    mapping: dict[Any, int],
    labels: list[str],
    monthly_psi: list[dict[str, Any]],
    metrics_backend: str,
) -> list[dict[str, Any]]:
    """Calculate the reference workbook's per-month KS/AUC/PSI/Lift table."""
    series = data.get_column(feature)
    numeric = _is_numeric(series)
    values = series.to_list()
    targets = [int(value) for value in data.get_column(target_col).to_list()]
    months = data.get_column(month_col).cast(pl.String).to_list()
    # A feature's direction is fixed globally so month-to-month Lift remains comparable.
    global_pairs = [
        (float(value), target)
        for value, target in zip(values, targets)
        if value is not None and math.isfinite(float(value))
    ] if numeric else [
        (float(mapping[value]), target)
        for value, target in zip(values, targets)
        if value is not None and value in mapping
    ]
    # _rank_auc is direction agnostic; use raw AUC to decide which end is risky.
    raw_global_auc = None
    if len(global_pairs) and len({target for _, target in global_pairs}) > 1:
        raw_global_auc = _raw_auc(
            [score for score, _ in global_pairs], [target for _, target in global_pairs]
        )
    descending = raw_global_auc is None or raw_global_auc >= 0.5
    psi_lookup = {(row["feature"], str(row["event_month"])): row["psi"] for row in monthly_psi}
    output: list[dict[str, Any]] = []
    for month in sorted({str(value) for value in months if value is not None}):
        month_pairs = [
            (float(value) if numeric else float(mapping[value]), target)
            for value, target, event_month in zip(values, targets, months)
            if event_month is not None and str(event_month) == month
            and value is not None and (not numeric or math.isfinite(float(value)))
            and (numeric or value in mapping)
        ]
        month_scores = [score for score, _ in month_pairs]
        month_targets = [target for _, target in month_pairs]
        sample_count = len(month_targets)
        bad_count = sum(month_targets)
        good_count = sample_count - bad_count
        raw_auc = _raw_auc(month_scores, month_targets) if month_scores else None
        auc = max(raw_auc, 1 - raw_auc) if raw_auc is not None else None
        ks = _metric_ks(month_scores, month_targets, metrics_backend)
        ranked = sorted(month_pairs, key=lambda item: item[0], reverse=descending)
        top_count = max(1, math.ceil(0.1 * sample_count)) if sample_count else 0
        top_bad_rate = (
            sum(target for _, target in ranked[:top_count]) / top_count
            if top_count else None
        )
        bad_rate = bad_count / sample_count if sample_count else None
        output.append(
            {
                "feature": feature,
                "event_month": month,
                "ks": ks,
                "auc": auc,
                "psi": float(psi_lookup.get((feature, month), 0.0) or 0.0),
                "lift_10": (
                    top_bad_rate / bad_rate
                    if top_bad_rate is not None and bad_rate
                    else None
                ),
                "sample_count": sample_count,
                "bad_count": bad_count,
                "good_count": good_count,
                "bad_rate": bad_rate,
                "sample_rate": sample_count / data.height if data.height else 0.0,
                "direction": "descending" if descending else "ascending",
            }
        )
    return output


def _raw_auc(values: list[float], targets: list[int]) -> float | None:
    """AUC before direction adjustment, used to orient Lift consistently."""
    bad_count = sum(targets)
    good_count = len(targets) - bad_count
    if not values or not bad_count or not good_count:
        return None
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    rank_sum = 0.0
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(targets[index] for index, _ in indexed[start:end])
        start = end
    return (rank_sum - bad_count * (bad_count + 1) / 2) / (bad_count * good_count)


def _ks_bucket_detail(
    data: pl.DataFrame,
    feature: str,
    target_col: str,
    mapping: dict[Any, int],
    bin_count: int,
    metrics_backend: str,
    method: str,
) -> list[dict[str, Any]]:
    """Produce a Toad-compatible ten-bucket style table for each feature."""
    series = data.get_column(feature)
    numeric = _is_numeric(series)
    pairs = [
        (float(value) if numeric else float(mapping[value]), int(target))
        for value, target in zip(series.to_list(), data.get_column(target_col).to_list())
        if value is not None and (not numeric or math.isfinite(float(value))) and (numeric or value in mapping)
    ]
    if not pairs:
        return []
    if metrics_backend == "toad":
        toad_rows = toad_ks_bucket(
            [score for score, _ in pairs],
            [target for _, target in pairs],
            bucket=bin_count,
            method=method,
        )
        if toad_rows:
            return [
                {
                    "feature": feature,
                    "bucket": index,
                    "score_min": row.get("min"),
                    "score_max": row.get("max"),
                    "bad_count": row.get("bads"),
                    "good_count": row.get("goods"),
                    "sample_count": row.get("total"),
                    "bad_rate": row.get("bad_rate"),
                    "good_rate": row.get("good_rate"),
                    "bad_distribution": row.get("bad_prop"),
                    "good_distribution": row.get("good_prop"),
                    "cumulative_bad_capture": row.get("cum_bads_prop"),
                    "cumulative_good_capture": row.get("cum_goods_prop"),
                    "cumulative_sample_rate": row.get("cum_total_prop"),
                    "ks": row.get("ks"),
                    "lift": row.get("lift"),
                    "cumulative_lift": row.get("cum_lift"),
                    "metrics_backend": "toad",
                }
                for index, row in enumerate(toad_rows, start=1)
            ]
    pairs.sort(key=lambda item: item[0], reverse=True)
    bucket_size = max(1, math.ceil(len(pairs) / bin_count))
    total_bad = sum(target for _, target in pairs)
    total_good = len(pairs) - total_bad
    total = len(pairs)
    bad_seen = good_seen = 0
    cumulative_count = 0
    rows: list[dict[str, Any]] = []
    for index in range(0, len(pairs), bucket_size):
        bucket_pairs = pairs[index : index + bucket_size]
        scores = [score for score, _ in bucket_pairs]
        bad_count = sum(target for _, target in bucket_pairs)
        good_count = len(bucket_pairs) - bad_count
        bad_seen += bad_count
        good_seen += good_count
        cumulative_count += len(bucket_pairs)
        sample_rate = len(bucket_pairs) / total
        bad_capture = bad_seen / total_bad if total_bad else None
        good_capture = good_seen / total_good if total_good else None
        ks = (
            abs(bad_capture - good_capture)
            if bad_capture is not None and good_capture is not None
            else None
        )
        bad_rate = bad_count / len(bucket_pairs) if bucket_pairs else None
        overall_bad_rate = total_bad / total if total else None
        rows.append(
            {
                "feature": feature,
                "bucket": len(rows) + 1,
                "score_min": min(scores),
                "score_max": max(scores),
                "bad_count": bad_count,
                "good_count": good_count,
                "sample_count": len(bucket_pairs),
                "bad_rate": bad_rate,
                "good_rate": good_count / len(bucket_pairs) if bucket_pairs else None,
                "bad_distribution": bad_count / total_bad if total_bad else None,
                "good_distribution": good_count / total_good if total_good else None,
                "cumulative_bad_capture": bad_capture,
                "cumulative_good_capture": good_capture,
                "cumulative_sample_rate": cumulative_count / total,
                "ks": ks,
                "lift": bad_rate / overall_bad_rate if bad_rate is not None and overall_bad_rate else None,
                "cumulative_lift": bad_capture / (cumulative_count / total) if bad_capture is not None and cumulative_count else None,
                "metrics_backend": metrics_backend,
            }
        )
    return rows


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for original_index, _ in indexed[start:end]:
            ranks[original_index] = average_rank
        start = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
    )
    right_denominator = math.sqrt(
        sum((value - right_mean) ** 2 for value in right)
    )
    return (
        numerator / (left_denominator * right_denominator)
        if left_denominator and right_denominator
        else 0.0
    )


def _correlation_pairs(
    data: pl.DataFrame,
    features: tuple[str, ...],
    method: str,
) -> list[dict[str, Any]]:
    numeric_features = [feature for feature in features if _is_numeric(data.get_column(feature))]
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(numeric_features):
        for right in numeric_features[left_index + 1 :]:
            pairs = [
                (float(x), float(y))
                for x, y in zip(data.get_column(left).to_list(), data.get_column(right).to_list())
                if x is not None and y is not None
            ]
            if len(pairs) < 2:
                continue
            x_values, y_values = zip(*pairs)
            if method == "spearman":
                x_values, y_values = _ranks(x_values), _ranks(y_values)
            correlation = _pearson(x_values, y_values)
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "method": method,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "pair_count": len(pairs),
                }
            )
    return sorted(rows, key=lambda row: row["abs_correlation"], reverse=True)


def build_eda_analysis(
    data: pl.DataFrame,
    contract: ValidatedDataContract,
    *,
    month_col: str = "event_month",
    bin_count: int = 10,
    ks_bucket: int | None = None,
    ks_method: str = "quantile",
    baseline_month: str | None = None,
    max_categories: int = 20,
    correlation_method: str = "pearson",
    metrics_backend: str = "toad",
) -> EdaAnalysisResult:
    """Build detailed feature, stability, and correlation EDA tables."""
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")
    if month_col not in data.columns:
        raise ValueError(f"EDA requires month column '{month_col}'")
    metrics_backend = resolve_backend(metrics_backend)
    if ks_method not in {"quantile", "step"}:
        raise ValueError("ks_method must be 'quantile' or 'step'")
    ks_bucket = int(ks_bucket or bin_count)
    if ks_bucket < 2:
        raise ValueError("ks_bucket must be at least 2")
    logger.info("Building EDA analysis: rows=%d, features=%d", data.height, len(contract.feature_cols))
    overview_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    psi_rows: list[dict[str, Any]] = []
    monthly_discrimination_rows: list[dict[str, Any]] = []
    ks_bucket_rows: list[dict[str, Any]] = []
    available_months = sorted(data.get_column(month_col).drop_nulls().cast(pl.String).unique().to_list())
    if not available_months:
        raise ValueError("EDA requires at least one non-null event_month")
    if baseline_month is None:
        baseline_month = available_months[0]
    elif str(baseline_month) not in available_months:
        raise ValueError(
            f"EDA PSI baseline month {baseline_month!r} is not available; "
            f"choose one of {available_months}"
        )
    else:
        baseline_month = str(baseline_month)
    for feature in contract.feature_cols:
        feature_details, metrics, mapping, labels = _bin_feature(
            data,
            feature,
            contract.contract.target_col,
            bin_count=bin_count,
            max_categories=max_categories,
            metrics_backend=metrics_backend,
        )
        details.extend(feature_details)
        overview_rows.append({"feature": feature, "metrics_backend": metrics_backend, **metrics})
        psi_rows.extend(
            _feature_psi(
                data,
                feature,
                month_col,
                mapping,
                labels,
                baseline_month,
                metrics_backend,
            )
        )
        # These two tables mirror the reference EDA workbook: monthly
        # discrimination and the KS/Lift score buckets.  They reuse the same
        # fitted bins and Toad-backed metrics as the headline feature table.
        monthly_discrimination_rows.extend(
            _monthly_discrimination(
                data,
                feature,
                contract.contract.target_col,
                month_col,
                mapping,
                labels,
                psi_rows,
                metrics_backend,
            )
        )
        ks_bucket_rows.extend(
            _ks_bucket_detail(
                data,
                feature,
                contract.contract.target_col,
                mapping,
                ks_bucket,
                metrics_backend,
                ks_method,
            )
        )

    monthly_sample = (
        data.group_by(month_col)
        .agg(
            pl.len().alias("sample_count"),
            pl.col(contract.contract.target_col).sum().alias("bad_count"),
        )
        .with_columns(
            (pl.col("sample_count") - pl.col("bad_count")).alias("good_count"),
            (pl.col("bad_count") / pl.col("sample_count")).alias("bad_rate"),
        )
        .sort(month_col)
    )
    result = EdaAnalysisResult(
        univariate_overview=pl.DataFrame(overview_rows).sort("iv", descending=True),
        binning_detail=pl.DataFrame(details),
        monthly_sample=monthly_sample,
        monthly_psi=pl.DataFrame(psi_rows).sort(["feature", "event_month"]),
        correlation_pairs=pl.DataFrame(
            _correlation_pairs(data, contract.feature_cols, correlation_method)
        ),
        monthly_discrimination=pl.DataFrame(monthly_discrimination_rows),
        ks_bucket=pl.DataFrame(ks_bucket_rows),
    )
    logger.info("EDA analysis completed: baseline_month=%s", baseline_month)
    return result
