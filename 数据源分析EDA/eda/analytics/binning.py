"""卡方分箱统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import toad
from tqdm.auto import tqdm

from eda.data import get_var_list


def binning_stats_chi(
    df: pd.DataFrame,
    target: str = "y_flag",
    exclude_cols: list[str] | None = None,
    min_samples: float = 0.05,
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target)

    # 分离全空列和非全空列
    all_empty_cols = [col for col in features if df[col].isna().all()]
    valid_features = [col for col in features if col not in all_empty_cols]

    combiner = toad.transform.Combiner()
    if valid_features:
        combiner.fit(
            df[valid_features],
            df[target],
            method="chi",
            min_samples=min_samples,
            empty_separate="separate",
        )
    bins_dict = combiner.export()

    label_maps: dict[str, dict[int, str]] = {}
    for var in valid_features:
        cut_points = [c for c in bins_dict.get(var, []) if not pd.isna(c)]
        bins = [-np.inf] + cut_points + [np.inf]
        labels = []
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            if left == -np.inf:
                label = f"[-inf, {right})"
            elif right == np.inf:
                label = f"[{left}, inf)"
            else:
                label = f"[{left}, {right})"
            labels.append(label)
        label_maps[var] = dict(zip(range(len(labels)), labels))

    # 全空列的分箱标签映射
    for var in all_empty_cols:
        label_maps[var] = {0: "nan"}

    df_binned = combiner.transform(df[valid_features], labels=False) if valid_features else pd.DataFrame()
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    total_samples = len(df)
    all_bins: list[pd.DataFrame] = []

    for var in tqdm(features, desc="卡方分箱", leave=False):
        # 全空列特殊处理
        if var in all_empty_cols:
            good_count = (df[target] == 0).sum()
            bad_count = (df[target] == 1).sum()
            total_count = len(df)
            badrate = bad_count / total_count if total_count > 0 else 0.0

            grouped = pd.DataFrame({
                "bin_id": [-1],
                "good": [good_count],
                "bad": [bad_count],
                "total": [total_count],
                "badrate": [badrate],
                "prop": [1.0],
                "y_prop": [1.0] if total_bad > 0 and total_count > 0 else [0.0],
                "n_prop": [1.0] if total_good > 0 and total_count > 0 else [0.0],
                "woe": [0.0],
                "iv_bin": [0.0],
                "iv": [0.0],
                "ks": [0.0],
            })
            grouped["groups"] = "nan"
            grouped["var"] = var
            all_bins.append(
                grouped[
                    [
                        "var",
                        "groups",
                        "good",
                        "bad",
                        "total",
                        "badrate",
                        "prop",
                        "y_prop",
                        "n_prop",
                        "woe",
                        "iv_bin",
                        "iv",
                        "ks",
                    ]
                ]
            )
            continue

        temp = pd.DataFrame({"bin_id": df_binned[var], target: df[target]})
        temp["bin_id"] = temp["bin_id"].fillna(-1)

        grouped = (
            temp.groupby("bin_id")[target]
            .agg(
                good=lambda x: (x == 0).sum(),
                bad=lambda x: (x == 1).sum(),
                total="count",
            )
            .reset_index()
        )
        grouped = grouped[grouped["total"] > 0]

        grouped["badrate"] = grouped["bad"] / grouped["total"]
        grouped["prop"] = grouped["total"] / total_samples
        grouped["y_prop"] = grouped["bad"] / total_bad if total_bad > 0 else 0
        grouped["n_prop"] = grouped["good"] / total_good if total_good > 0 else 0

        eps = 1e-10
        yp = grouped["y_prop"].replace(0, eps)
        np_val = grouped["n_prop"].replace(0, eps)
        grouped["woe"] = np.log(yp / np_val)
        grouped["iv_bin"] = (grouped["y_prop"] - grouped["n_prop"]) * grouped["woe"]

        grouped = grouped.sort_values("bin_id")
        grouped["iv"] = grouped["iv_bin"].cumsum()

        total_good_group = grouped["good"].sum()
        total_bad_group = grouped["bad"].sum()
        grouped["cum_good"] = (
            grouped["good"].cumsum() / total_good_group if total_good_group > 0 else 0
        )
        grouped["cum_bad"] = (
            grouped["bad"].cumsum() / total_bad_group if total_bad_group > 0 else 0
        )
        grouped["ks_diff"] = np.abs(grouped["cum_good"] - grouped["cum_bad"])
        grouped["ks"] = grouped["ks_diff"].cummax()

        map_dict = label_maps[var]
        grouped["groups"] = grouped["bin_id"].apply(
            lambda x: map_dict.get(x, "nan") if x >= 0 else "nan"
        )
        grouped = grouped.drop(
            columns=["bin_id", "cum_good", "cum_bad", "ks_diff"], errors="ignore"
        )
        grouped["var"] = var
        all_bins.append(
            grouped[
                [
                    "var",
                    "groups",
                    "good",
                    "bad",
                    "total",
                    "badrate",
                    "prop",
                    "y_prop",
                    "n_prop",
                    "woe",
                    "iv_bin",
                    "iv",
                    "ks",
                ]
            ]
        )

    result = pd.concat(all_bins, ignore_index=True)
    return result.rename(
        columns={
            "var": "变量名",
            "groups": "分箱区间",
            "good": "好样本",
            "bad": "坏样本",
            "total": "总样本",
            "badrate": "坏账率",
            "prop": "样本占比",
            "y_prop": "坏样本占比",
            "n_prop": "好样本占比",
            "woe": "WOE值",
            "iv_bin": "分箱IV值",
            "iv": "累计IV值",
            "ks": "累计KS值",
        }
    )
