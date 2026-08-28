"""变量概览表（描述统计 + IV/KS）。"""

from __future__ import annotations

import pandas as pd
import toad
from tqdm.auto import tqdm

from eda.analytics._metrics import safe_ks
from eda.data import get_var_list


def first_table(
    df: pd.DataFrame,
    target: str = "y_flag",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target)

    # 识别全空列
    all_empty_cols = [col for col in features if df[col].isna().all()]

    desc = df[features].describe(percentiles=[0.01, 0.10, 0.50, 0.75, 0.90, 0.99]).T
    desc = desc[["mean", "std", "min", "1%", "10%", "50%", "75%", "90%", "99%", "max"]]

    # 全空列的统计值设为 "-"
    stat_cols = ["mean", "std", "min", "1%", "10%", "50%", "75%", "90%", "99%", "max"]
    for col in all_empty_cols:
        for stat_col in stat_cols:
            desc.loc[col, stat_col] = "-"

    info = (
        pd.DataFrame(
            {
                "type": df[features].dtypes,
                "size": df[features].count(),
                "missing": df[features].isnull().sum() / df.shape[0],
                "complete": 1 - df[features].isnull().sum() / df.shape[0],
            }
        )
        .reset_index()
        .rename(columns={"index": "var_name"})
    )

    quality = toad.quality(df[features + [target]], target=target, iv_only=False)
    ks_list = []
    for var in tqdm(quality.index, desc="计算KS值", leave=False):
        ks_val = safe_ks(df[[var, target]].dropna()[var], df[[var, target]].dropna()[target])
        ks_list.append(ks_val)
    quality["ks"] = ks_list
    quality = quality[["iv", "ks", "unique"]]

    final = info.merge(desc, left_on="var_name", right_index=True, how="left")
    final = final.merge(quality, left_on="var_name", right_index=True, how="left")
    final = final[
        [
            "var_name",
            "type",
            "missing",
            "complete",
            "unique",
            "iv",
            "ks",
            "mean",
            "min",
            "max",
            "std",
            "1%",
            "10%",
            "50%",
            "75%",
            "90%",
            "99%",
        ]
    ]
    final = final.rename(
        columns={
            "var_name": "变量名",
            "type": "类型",
            "missing": "缺失率",
            "complete": "完整率",
            "unique": "唯一值",
            "iv": "IV值",
            "ks": "KS值",
            "mean": "平均值",
            "min": "最小值",
            "max": "最大值",
            "std": "标准差",
            "1%": "1%分位",
            "10%": "10%分位",
            "50%": "50%分位",
            "75%": "75%分位",
            "90%": "90%分位",
            "99%": "99%分位",
        }
    ).sort_values("KS值", ascending=False)
    return final
