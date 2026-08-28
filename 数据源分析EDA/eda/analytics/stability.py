"""跨月 PSI 稳定性分析。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eda.analytics._metrics import safe_psi
from eda.data import get_var_list

logger = logging.getLogger(__name__)


def psi_against_base_toad(
    df: pd.DataFrame,
    target: str = "y_flag",
    exclude_cols: list[str] | None = None,
    date_col: str = "clean_date",
    yearmonth_col: str = "yearmonth",
    base_month: str | None = None,
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target, yearmonth_col=yearmonth_col)
    all_months = sorted(df[yearmonth_col].unique())

    if len(all_months) < 2:
        raise ValueError("至少需要2个月份才能计算PSI")

    if base_month is None:
        base_month = all_months[0]
    target_months = [m for m in all_months if m != base_month]

    base_data = df[df[yearmonth_col] == base_month]
    if base_data.empty:
        raise ValueError(f"基期 {base_month} 无数据")

    # 识别全空列
    all_empty_cols = set(col for col in features if df[col].isna().all())

    results: list[dict] = []
    for var in tqdm(features, desc="计算PSI", leave=False):
        expected = base_data[var].dropna()
        if expected.empty or len(expected) < 2:
            row: dict = {"var": var, base_month: 0.0}
            if var in all_empty_cols:
                # 全空列的PSI值设为0.0
                row.update({m: 0.0 for m in target_months})
            else:
                row.update({m: np.nan for m in target_months})
            results.append(row)
            continue

        row = {"var": var, base_month: 0.0}
        for month in target_months:
            actual = df[df[yearmonth_col] == month][var].dropna()
            if actual.empty or len(actual) < 2:
                row[month] = np.nan
            else:
                psi_val = safe_psi(expected, actual)
                if np.isnan(psi_val):
                    logger.warning("变量 %s 在月份 %s 计算PSI失败", var, month)
                row[month] = psi_val
        results.append(row)

    psi_df = pd.DataFrame(results)
    cols = ["var"] + [base_month] + target_months
    psi_df = psi_df[cols]

    psi_columns = [m for m in target_months if m in psi_df.columns]
    if psi_columns:
        psi_df["avg_psi"] = psi_df[psi_columns].mean(axis=1, skipna=True)
        psi_df["max_psi"] = psi_df[psi_columns].max(axis=1, skipna=True)
        psi_df["psi_stability"] = psi_df["avg_psi"].apply(
            lambda x: "-" if pd.isna(x) or x < 0.1 else ("! ! !" if x > 0.25 else "! !")
        )
        psi_df = psi_df.sort_values("avg_psi", ascending=False).reset_index(drop=True)

    rename_dict: dict = {"var": "变量名"}
    for col in psi_df.columns:
        if col not in ("var", "avg_psi", "max_psi", "psi_stability"):
            rename_dict[col] = str(col)
    rename_dict.update(
        {
            "avg_psi": "平均PSI",
            "max_psi": "最大PSI",
            "psi_stability": "稳定性评估",
        }
    )
    return psi_df.rename(columns=rename_dict)
