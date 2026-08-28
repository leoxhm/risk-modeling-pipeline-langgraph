"""Small, lazy adapter around Toad's IV, KS, PSI and KS-bucket implementations.

The rest of the project uses Polars, while Toad currently accepts NumPy/pandas
like arrays.  Keeping the conversion here makes the metric convention explicit
and keeps Toad optional for backwards-compatible environments.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from logger import get_logger


logger = get_logger(__name__)


def _load_toad() -> tuple[Any, Any]:
    try:
        from toad import metrics, stats
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'toad' package is required for metrics_backend='toad'. "
            "Install it with: pip install 'toad>=0.1.7,<0.2'"
        ) from exc
    return stats, metrics


def resolve_backend(backend: str = "toad") -> str:
    """Validate a backend and fall back to legacy only when Toad is absent."""
    if backend not in {"toad", "legacy"}:
        raise ValueError("metrics_backend must be 'toad' or 'legacy'")
    if backend == "legacy":
        return backend
    try:
        _load_toad()
    except RuntimeError:
        logger.warning(
            "Toad is not installed; falling back to the legacy metric backend."
        )
        return "legacy"
    return "toad"


def available_backends() -> tuple[str, ...]:
    """Return the metric backends available in this Python environment."""
    return ("legacy", "toad") if resolve_backend("toad") == "toad" else ("legacy",)


def _finite_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def toad_iv_from_bins(
    bins: Iterable[int],
    targets: Iterable[int],
) -> tuple[float | None, dict[int, float]]:
    """Calculate IV using Toad on an already-fitted, auditable bin mapping.

    Passing bin IDs (rather than re-binning the raw feature inside this helper)
    ensures the headline IV and the displayed binning detail use the same bins.
    """
    stats, _ = _load_toad()
    bin_values = np.asarray([int(value) for value in bins])
    target_values = np.asarray([int(value) for value in targets])
    if not len(bin_values) or len(bin_values) != len(target_values):
        return None, {}
    try:
        iv, sub = stats.IV(bin_values, target_values, return_sub=True)
    except (TypeError, ValueError, ZeroDivisionError):
        return None, {}
    sub_values = {
        int(key): float(value)
        for key, value in sub.items()
        if _finite_or_none(value) is not None
    }
    return _finite_or_none(iv), sub_values


def toad_ks(scores: Iterable[float], targets: Iterable[int]) -> float | None:
    """Calculate KS using ``toad.metrics.KS`` (two-sample KS statistic)."""
    _, metrics = _load_toad()
    score_values = np.asarray(list(scores), dtype=float)
    target_values = np.asarray([int(value) for value in targets])
    if not len(score_values) or len(score_values) != len(target_values):
        return None
    if len(np.unique(target_values)) < 2:
        return None
    try:
        return _finite_or_none(metrics.KS(score_values, target_values))
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
        return None


def toad_psi(
    test: Iterable[int | float],
    base: Iterable[int | float],
    *,
    support: Iterable[int | float] | None = None,
) -> float | None:
    """Calculate PSI using Toad, completing the shared bin support first.

    Toad's PSI implementation aligns value-count Series and intentionally does
    not add a zero-cell correction.  If a month has no observations in a bin,
    pandas alignment would otherwise produce NaN and Toad would return zero.
    Adding one support observation to both sides preserves the same categories
    and makes the result finite and auditable for monthly stability analysis.
    """
    _, metrics = _load_toad()
    test_values = list(test)
    base_values = list(base)
    if not test_values or not base_values:
        return None
    support_values = list(dict.fromkeys(support or [*test_values, *base_values]))
    # Use one copy of every possible bucket on both sides to avoid Toad's
    # missing-category/NaN behavior while keeping the correction negligible.
    test_augmented = [*test_values, *support_values]
    base_augmented = [*base_values, *support_values]
    try:
        return _finite_or_none(
            metrics.PSI(np.asarray(test_augmented), np.asarray(base_augmented))
        )
    except (TypeError, ValueError, IndexError):
        return None


def toad_ks_bucket(
    scores: Iterable[float],
    targets: Iterable[int],
    *,
    bucket: int = 10,
    method: str = "quantile",
) -> list[dict[str, Any]]:
    """Return Toad's KS-bucket/Lift detail as JSON-ready row dictionaries."""
    _, metrics = _load_toad()
    score_values = np.asarray(list(scores), dtype=float)
    target_values = np.asarray([int(value) for value in targets])
    if not len(score_values) or len(score_values) != len(target_values):
        return []
    if len(np.unique(target_values)) < 2:
        return []
    try:
        result = metrics.KS_bucket(
            score=score_values,
            target=target_values,
            bucket=int(bucket),
            method=method,
        )
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
        return []
    return [
        {
            str(key): (
                value.item() if hasattr(value, "item") else value
            )
            for key, value in row.items()
        }
        for row in result.to_dict(orient="records")
    ]
