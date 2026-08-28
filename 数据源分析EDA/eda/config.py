"""配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    input_path: str
    target_col: str = "y_flag"
    date_col: str = "clean_date"
    exclude_cols: list[str] = field(
        default_factory=lambda: ["map_key", "clean_date", "y_flag"]
    )


@dataclass
class AnalysisConfig:
    neg_corr_list: list[str] = field(default_factory=list)
    chi_min_samples: float = 0.05
    ks_bucket: int = 10
    ks_method: str = "quantile"
    psi_base_month: str | None = None


@dataclass
class OutputConfig:
    excel_path: str = "output/eda_report.xlsx"


@dataclass
class ExcelConfig:
    freeze_panes: bool = True
    autofilter: bool = True
    max_col_width: int = 42
    corr_col_width: int = 12
    corr_row_height: int = 60


@dataclass
class AppConfig:
    data: DataConfig
    analysis: AnalysisConfig
    output: OutputConfig
    excel: ExcelConfig
    config_dir: Path = field(default_factory=Path.cwd)

    def resolve_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return (self.config_dir / path).resolve()


def _merge_dataclass(cls: type, raw: dict[str, Any] | None) -> Any:
    if raw is None:
        raw = {}
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in fields}
    return cls(**filtered)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        data=_merge_dataclass(DataConfig, raw.get("data")),
        analysis=_merge_dataclass(AnalysisConfig, raw.get("analysis")),
        output=_merge_dataclass(OutputConfig, raw.get("output")),
        excel=_merge_dataclass(ExcelConfig, raw.get("excel")),
        config_dir=path.parent,
    )
