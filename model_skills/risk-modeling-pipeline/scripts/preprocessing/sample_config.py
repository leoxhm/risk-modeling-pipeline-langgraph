"""Validated configuration for sample diagnostics and treatment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class SampleConfigError(ValueError):
    """Raised when sample diagnostic or treatment configuration is unsafe."""


@dataclass(frozen=True)
class SampleDiagnosticThresholds:
    """Thresholds used only to classify deterministic sample findings."""

    imbalance_warning_minority_rate: float
    imbalance_critical_minority_rate: float
    latest_month_min_volume_ratio: float
    monthly_bad_rate_change_warning: float
    high_missing_row_rate: float


@dataclass(frozen=True)
class SampleTreatmentConfig:
    """User-confirmed actions for a later preprocessing/modeling stage."""

    duplicate_action: str
    missing_target_action: str
    all_null_feature_action: str
    high_missing_row_action: str
    incomplete_latest_month_action: str
    class_imbalance_action: str


@dataclass(frozen=True)
class SampleConfig:
    """Complete sample diagnosis and treatment configuration."""

    diagnostics: SampleDiagnosticThresholds
    treatment: SampleTreatmentConfig


def _mapping(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise SampleConfigError(f"Configuration section '{name}' must be a mapping")
    return value


def load_sample_config(path: str | Path) -> SampleConfig:
    """Load a sample configuration without inspecting or changing data."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Sample configuration does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        # Standalone node configs use the common envelope
        # {node_id, version, parameters: {diagnostics, treatment}}. Keep
        # accepting the legacy top-level shape for batch compatibility.
        if isinstance(raw.get("parameters"), dict):
            raw = raw["parameters"]
        diagnostics = _mapping(raw, "diagnostics")
        treatment = _mapping(raw, "treatment")
        config = SampleConfig(
            diagnostics=SampleDiagnosticThresholds(
                imbalance_warning_minority_rate=float(
                    diagnostics.get("imbalance_warning_minority_rate", 0.10)
                ),
                imbalance_critical_minority_rate=float(
                    diagnostics.get("imbalance_critical_minority_rate", 0.01)
                ),
                latest_month_min_volume_ratio=float(
                    diagnostics.get("latest_month_min_volume_ratio", 0.50)
                ),
                monthly_bad_rate_change_warning=float(
                    diagnostics.get("monthly_bad_rate_change_warning", 0.03)
                ),
                high_missing_row_rate=float(
                    diagnostics.get("high_missing_row_rate", 0.80)
                ),
            ),
            treatment=SampleTreatmentConfig(
                duplicate_action=str(treatment.get("duplicate_action", "error")),
                missing_target_action=str(
                    treatment.get("missing_target_action", "error")
                ),
                all_null_feature_action=str(
                    treatment.get("all_null_feature_action", "drop")
                ),
                high_missing_row_action=str(
                    treatment.get("high_missing_row_action", "keep")
                ),
                incomplete_latest_month_action=str(
                    treatment.get("incomplete_latest_month_action", "error")
                ),
                class_imbalance_action=str(
                    treatment.get("class_imbalance_action", "class_weight")
                ),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SampleConfigError(f"Invalid sample configuration: {source}") from exc

    thresholds = config.diagnostics
    if (
        not 0
        < thresholds.imbalance_critical_minority_rate
        <= thresholds.imbalance_warning_minority_rate
        < 0.5
    ):
        raise SampleConfigError(
            "Imbalance thresholds must satisfy 0 < critical <= warning < 0.5"
        )
    for name, value in (
        ("latest_month_min_volume_ratio", thresholds.latest_month_min_volume_ratio),
        ("monthly_bad_rate_change_warning", thresholds.monthly_bad_rate_change_warning),
        ("high_missing_row_rate", thresholds.high_missing_row_rate),
    ):
        if not 0 < value <= 1:
            raise SampleConfigError(f"diagnostics.{name} must be between 0 and 1")

    allowed_actions = {
        "duplicate_action": {"error", "keep_first", "keep_last"},
        "missing_target_action": {"error", "drop"},
        "all_null_feature_action": {"error", "drop"},
        "high_missing_row_action": {"keep", "drop"},
        "incomplete_latest_month_action": {"error", "keep", "exclude"},
        "class_imbalance_action": {"none", "class_weight"},
    }
    for name, allowed in allowed_actions.items():
        value = getattr(config.treatment, name)
        if value not in allowed:
            raise SampleConfigError(
                f"treatment.{name} must be one of {sorted(allowed)}"
            )
    return config
