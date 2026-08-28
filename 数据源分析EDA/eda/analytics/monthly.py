"""变量月度区分度统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eda.analytics._metrics import safe_auc, safe_ks, safe_psi
from eda.data import get_var_list


def variable_monthly_stats(
    df: pd.DataFrame,
    target: str = "y_flag",
    yearmonth_col: str = "yearmonth",
    base_month: str | None = None,
    neg_corr_list: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    date_col: str = "clean_date",
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target, yearmonth_col=yearmonth_col)
    neg_corr_set = set(neg_corr_list or [])

    months = sorted(df[yearmonth_col].unique())
    if base_month is None:
        base_month = months[0]

    monthly_stats = df.groupby(yearmonth_col)[target].agg(total="count", bad="sum").reset_index()
    monthly_stats["good"] = monthly_stats["total"] - monthly_stats["bad"]
    monthly_stats["badrate"] = monthly_stats["bad"] / monthly_stats["total"]
    all_total = monthly_stats["total"].sum()
    monthly_stats["prop"] = monthly_stats["total"] / all_total

    results: list[dict] = []
    for var in tqdm(features, desc="月度统计", leave=False):
        for month in months:
            month_data = df[df[yearmonth_col] == month][[var, target]].dropna()

            if len(month_data) < 2 or month_data[var].nunique() < 2:
                results.append(
                    {
                        "var": var,
                        yearmonth_col: month,
                        "ym_ks": "-",
                        "ym_auc": "-",
                        "ym_psi": 0,
                        "lift_10%": "-",
                    }
                )
                continue

            n_bad = int(month_data[target].sum())
            n_good = len(month_data) - n_bad

            if n_bad == 0 or n_good == 0:
                ks_val = "-"
                auc_val = "-"
            else:
                ks_val = safe_ks(month_data[var], month_data[target], default="-")
                auc_val = safe_auc(month_data[target], month_data[var])

            if month == base_month:
                psi_val = 0.0
            else:
                base_data = df[df[yearmonth_col] == base_month][[var, target]].dropna()
                if len(base_data) < 2:
                    psi_val = "-"
                else:
                    psi_val = safe_psi(base_data[var], month_data[var])

            if n_bad == 0 or n_good == 0:
                lift_val = "-"
            else:
                ascending = var in neg_corr_set
                sorted_month = month_data.sort_values(var, ascending=ascending)
                n_top = max(1, int(np.ceil(0.1 * len(sorted_month))))
                top_data = sorted_month.head(n_top)
                total_bad_rate = month_data[target].mean()
                lift_val = (
                    top_data[target].mean() / total_bad_rate if total_bad_rate > 0 else "-"
                )

            results.append(
                {
                    "var": var,
                    yearmonth_col: month,
                    "ym_ks": ks_val,
                    "ym_auc": auc_val,
                    "ym_psi": psi_val,
                    "lift_10%": lift_val,
                }
            )

    result_df = pd.DataFrame(results)
    result_df = result_df.merge(monthly_stats, on=yearmonth_col, how="left")
    base_cols = ["var", yearmonth_col, "ym_ks", "ym_auc", "ym_psi", "lift_10%"]
    extra_cols = ["good", "bad", "total", "badrate", "prop"]
    result_df = result_df[base_cols + extra_cols]
    return result_df.rename(
        columns={
            "var": "变量名",
            yearmonth_col: "年月",
            "ym_ks": "KS值",
            "ym_auc": "AUC值",
            "ym_psi": "PSI值",
            "lift_10%": "Lift前10%",
            "good": "好样本数",
            "bad": "坏样本数",
            "total": "总样本数",
            "badrate": "坏账率",
            "prop": "样本占比",
        }
    )
