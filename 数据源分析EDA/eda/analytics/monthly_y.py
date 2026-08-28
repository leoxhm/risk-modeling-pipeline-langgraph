"""月度样本基础统计。"""

from __future__ import annotations

import pandas as pd
import toad


def monthly_y_distribution(
    df: pd.DataFrame,
    target: str = "y_flag",
    yearmonth_col: str = "yearmonth",
) -> pd.DataFrame:
    """计算月度样本分布与坏账率统计（以第一个月为基期计算PSI）。"""
    all_months = sorted(df[yearmonth_col].unique())
    if len(all_months) < 2:
        raise ValueError("至少需要2个月份才能计算样本统计")

    # 计算基础统计
    monthly = df.groupby(yearmonth_col)[target].agg(
        total="count",
        bads="sum",
        goods=lambda x: (x == 0).sum(),
    ).reset_index()

    monthly["bad_rate"] = monthly["bads"] / monthly["total"]
    monthly["good_rate"] = monthly["goods"] / monthly["total"]

    # 计算 PSI（以第一个月为基期）
    base_month = all_months[0]
    base_dist = df[df[yearmonth_col] == base_month][target]

    def calc_psi(month: pd.Series) -> float:
        if month == base_month:
            return 0.0
        actual = df[df[yearmonth_col] == month][target]
        return toad.metrics.PSI(base_dist, actual)

    monthly["psi"] = monthly[yearmonth_col].apply(calc_psi)

    return monthly.rename(
        columns={
            yearmonth_col: "年月",
            "total": "样本总量",
            "bads": "坏样本数",
            "goods": "好样本数",
            "bad_rate": "坏账率",
            "good_rate": "好账率",
            "psi": "PSI值",
        }
    )