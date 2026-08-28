"""数据加载与清洗。"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np
from eda.config import DataConfig


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _make_yearmonth_col(df: pd.DataFrame, date_col: str) -> str:
    """根据date_col生成唯一的yearmonth列名，避免与现有列冲突。"""
    base = f"{date_col}_yearmonth"
    col = base
    idx = 1
    while col in df.columns:
        col = f"{base}_{idx}"
        idx += 1
    return col


def data_time_change_vectorized(date_series: pd.Series) -> pd.Series:
    date_str = date_series.astype(str).str.replace(r'[/\-年月\s]', '', regex=True)
    mask_6 = (date_str.str.len() == 6) & date_str.str.isdigit()
    date_str = date_str.where(~mask_6, date_str + '01')
    date_str = date_str.str[:8]
    result = date_str.str[:6]
    result = result.where(result.str.match(r'^\d{6}$'), np.nan)
    return result

def df_clean(df: pd.DataFrame, date_col: str = "clean_date") -> tuple[pd.DataFrame, str, list[str]]:
    """
    数据清洗并生成yearmonth列。
    返回: (清洗后的DataFrame, yearmonth列名, exclude_cols(含新生成的yearmonth列))
    """
    df = df.drop_duplicates().copy()
    yearmonth_col = _make_yearmonth_col(df, date_col)
    df[yearmonth_col] = data_time_change_vectorized(df[date_col])
    return df, yearmonth_col, [yearmonth_col]


def get_var_list(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    target_col: str = "y_flag",
    yearmonth_col: str | None = None,
) -> list[str]:
    if exclude_cols is None:
        exclude_cols = ["map_key", "clean_date", target_col]
    if yearmonth_col and yearmonth_col not in exclude_cols:
        exclude_cols.append(yearmonth_col)
    return [c for c in df.columns if c not in exclude_cols]


def load_and_clean(cfg: DataConfig, base_dir: Path | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """
    加载并清洗数据，返回(DataFrame, yearmonth列名, yearmonth_exclude列表)。
    """
    if base_dir is None:
        base_dir = Path.cwd()
    input_path = Path(cfg.input_path)
    if not input_path.is_absolute():
        input_path = (base_dir / input_path).resolve()
    df = load_csv(input_path)
    return df_clean(df, date_col=cfg.date_col)
