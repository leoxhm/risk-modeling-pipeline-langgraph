"""Deterministic binary-model validation metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import polars as pl

from metrics import resolve_backend, toad_ks, toad_psi


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics and decile lift detail for one scored sample."""

    auc: float | None
    ks: float | None
    lift: pl.DataFrame
    backend: str = "toad"


def _rank_auc(labels: list[int], scores: list[float]) -> float | None:
    bad_count = sum(labels)
    good_count = len(labels) - bad_count
    if not bad_count or not good_count:
        return None
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for original_index, _ in indexed[start:end]:
            ranks[original_index] = average_rank
        start = end
    bad_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (bad_rank_sum - bad_count * (bad_count + 1) / 2) / (bad_count * good_count)


def _ks(labels: list[int], scores: list[float]) -> float | None:
    bad_total = sum(labels)
    good_total = len(labels) - bad_total
    if not bad_total or not good_total:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    bad_cumulative = 0
    good_cumulative = 0
    maximum = 0.0
    for _, label in ordered:
        bad_cumulative += int(label)
        good_cumulative += 1 - int(label)
        maximum = max(maximum, abs(bad_cumulative / bad_total - good_cumulative / good_total))
    return maximum


def calculate_auc_ks(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    metrics_backend: str = "toad",
) -> tuple[float | None, float | None]:
    """Return AUC and KS without constructing the lift table."""
    label_values = [int(value) for value in labels]
    score_values = [float(value) for value in scores]
    resolved_backend = resolve_backend(metrics_backend)
    auc = _rank_auc(label_values, score_values)
    ks = (
        toad_ks(score_values, label_values)
        if resolved_backend == "toad"
        else _ks(label_values, score_values)
    )
    return auc, ks


def build_lift_table(labels: Iterable[int], scores: Iterable[float], bins: int = 10) -> pl.DataFrame:
    """Build equal-population score buckets, from highest to lowest risk."""
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    if not pairs:
        return pl.DataFrame(schema={"bucket": pl.Int64})
    overall_bad_rate = sum(label for _, label in pairs) / len(pairs)
    rows: list[dict[str, float | int]] = []
    for bucket in range(1, bins + 1):
        start = math.floor((bucket - 1) * len(pairs) / bins)
        end = math.floor(bucket * len(pairs) / bins)
        bucket_pairs = pairs[start:end]
        if not bucket_pairs:
            continue
        sample_count = len(bucket_pairs)
        bad_count = sum(label for _, label in bucket_pairs)
        bad_rate = bad_count / sample_count
        rows.append(
            {
                "bucket": bucket,
                "score_min": min(score for score, _ in bucket_pairs),
                "score_max": max(score for score, _ in bucket_pairs),
                "sample_count": sample_count,
                "bad_count": bad_count,
                "bad_rate": bad_rate,
                "lift": bad_rate / overall_bad_rate if overall_bad_rate else None,
            }
        )
    return pl.DataFrame(rows)


def evaluate_binary_model(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    metrics_backend: str = "toad",
) -> EvaluationResult:
    """Calculate AUC, KS, and decile lift using the selected metric backend."""
    label_values = [int(value) for value in labels]
    score_values = [float(value) for value in scores]
    resolved_backend = resolve_backend(metrics_backend)
    auc, ks = calculate_auc_ks(
        label_values,
        score_values,
        metrics_backend=resolved_backend,
    )
    return EvaluationResult(
        auc=auc,
        ks=ks,
        lift=build_lift_table(label_values, score_values),
        backend=resolved_backend,
    )


def population_stability_index(
    reference: Iterable[float],
    actual: Iterable[float],
    bins: int = 10,
    *,
    metrics_backend: str = "toad",
) -> float | None:
    """Calculate score PSI using train-score quantile bins as the baseline."""
    reference_values = sorted(float(value) for value in reference)
    actual_values = [float(value) for value in actual]
    if not reference_values or not actual_values:
        return None
    edges = sorted(
        {
            reference_values[min(len(reference_values) - 1, round(index * (len(reference_values) - 1) / bins))]
            for index in range(1, bins)
        }
    )
    def bucket(value: float) -> int:
        return sum(value > edge for edge in edges)
    reference_bucket_values = [bucket(value) for value in reference_values]
    actual_bucket_values = [bucket(value) for value in actual_values]
    if resolve_backend(metrics_backend) == "toad":
        return toad_psi(
            actual_bucket_values,
            reference_bucket_values,
            support=range(len(edges) + 1),
        )

    ref_counts = [0] * (len(edges) + 1)
    actual_counts = [0] * (len(edges) + 1)
    for value in reference_bucket_values:
        ref_counts[value] += 1
    for value in actual_bucket_values:
        actual_counts[value] += 1
    epsilon = 0.5
    psi = 0.0
    for ref_count, actual_count in zip(ref_counts, actual_counts):
        ref_share = (ref_count + epsilon) / (len(reference_values) + epsilon * len(ref_counts))
        actual_share = (actual_count + epsilon) / (len(actual_values) + epsilon * len(actual_counts))
        psi += (actual_share - ref_share) * math.log(actual_share / ref_share)
    return psi
