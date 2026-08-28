"""KS 分位数分箱统计。"""

from __future__ import annotations

import pandas as pd
import toad
from tqdm.auto import tqdm

from eda.data import get_var_list


def ks_bucket_all_vars(
    df: pd.DataFrame,
    target: str = "y_flag",
    exclude_cols: list[str] | None = None,
    bucket: int = 10,
    method: str = "quantile",
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target)
    features = [f for f in features if f != target]

    # 识别全空列
    all_empty_cols = [col for col in features if df[col].isna().all()]
    valid_features = [f for f in features if f not in all_empty_cols]

    all_parts: list[pd.DataFrame] = []

    for var in tqdm(features, desc="KS分箱", leave=False):
        # 全空列特殊处理
        if var in all_empty_cols:
            total_good = (df[target] == 0).sum()
            total_bad = (df[target] == 1).sum()
            total = len(df)
            bad_rate = total_bad / total if total > 0 else 0.0

            empty_df = pd.DataFrame({
                "var": [var],
                "min": ["-"],
                "max": ["-"],
                "bads": [total_bad],
                "goods": [total_good],
                "total": [total],
                "bad_rate": [bad_rate],
                "good_rate": [1 - bad_rate],
                "odds": [total_good / total_bad if total_bad > 0 else "-"],
                "bad_prop": [1.0],
                "good_prop": [1.0],
                "total_prop": [1.0],
                "cum_bad_rate": [bad_rate],
                "cum_bad_rate_rev": [bad_rate],
                "cum_bads_prop": [1.0],
                "cum_bads_prop_rev": [1.0],
                "cum_goods_prop": [1.0],
                "cum_goods_prop_rev": [1.0],
                "cum_total_prop": [1.0],
                "cum_total_prop_rev": [1.0],
                "ks": [0.0],
                "lift": ["-"],
                "cum_lift": ["-"],
            })
            all_parts.append(empty_df)
            continue

        temp = df[[var, target]].dropna()
        if len(temp) < 2:
            continue
        ks_df = toad.metrics.KS_bucket(
            score=temp[var], target=temp[target], bucket=bucket, method=method
        )
        ks_df["var"] = var
        cols = ["var"] + [c for c in ks_df.columns if c != "var"]
        all_parts.append(ks_df[cols])

    if not all_parts:
        return pd.DataFrame()

    result_df = pd.concat(all_parts, ignore_index=True)

    # 分箱区间：全空列显示nan，其他正常计算
    def format_bucket(row):
        if pd.isna(row["min"]) and pd.isna(row["max"]):
            return "nan"
        try:
            return f"[{row['min']:.2f}, {row['max']:.2f})"
        except (ValueError, TypeError):
            return f"[{row['min']}, {row['max']})"

    result_df["分箱区间"] = result_df.apply(format_bucket, axis=1)
    cols = ["var", "分箱区间"] + [
        c for c in result_df.columns if c not in ("var", "分箱区间")
    ]
    result_df = result_df[cols]
    return result_df.rename(
        columns={
            "var": "变量名",
            "min": "最小值",
            "max": "最大值",
            "bads": "坏样本数",
            "goods": "好样本数",
            "total": "总样本数",
            "bad_rate": "坏账率",
            "good_rate": "好账率",
            "odds": "好坏比",
            "bad_prop": "坏样本占比",
            "good_prop": "好样本占比",
            "total_prop": "总样本占比",
            "cum_bad_rate": "累计坏账率",
            "cum_bad_rate_rev": "反向累计坏账率",
            "cum_bads_prop": "累计坏样本占比",
            "cum_bads_prop_rev": "反向累计坏样本占比",
            "cum_goods_prop": "累计好样本占比",
            "cum_goods_prop_rev": "反向累计好样本占比",
            "cum_total_prop": "累计样本占比",
            "cum_total_prop_rev": "反向累计样本占比",
            "ks": "KS值",
            "lift": "Lift值",
            "cum_lift": "累计Lift值",
        }
    )
