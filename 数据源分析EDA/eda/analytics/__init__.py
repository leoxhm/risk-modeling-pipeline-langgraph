"""分析模块入口。"""

from __future__ import annotations

import logging
from pathlib import Path

from flask.scaffold import F
import pandas as pd

from eda.analytics.binning import binning_stats_chi
from eda.analytics.correlation import corr_matrix
from eda.analytics.ks_bucket import ks_bucket_all_vars
from eda.analytics.monthly import variable_monthly_stats
from eda.analytics.monthly_y import monthly_y_distribution
from eda.analytics.overview import first_table
from eda.analytics.stability import psi_against_base_toad
from eda.config import AppConfig
from eda.plots import plot_features_separately

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

SHEET_SAMPLE = "1. 样本基础统计"
SHEET_OVERVIEW = "2. 变量概览"
SHEET_BINNING = "3. 变量卡方分箱"
SHEET_PSI = "4. 变量PSI稳定性"
SHEET_CORR = "5. 变量相关性"
SHEET_MONTHLY = "6. 变量月度区分度"
SHEET_KS = "7. 变量KS十分箱"
SHEET_PLOTS = "8. 变量分布图"


def run_all(
    df: pd.DataFrame,
    cfg: AppConfig,
    yearmonth_col: str,
    yearmonth_exclude: list[str],
    plots_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], Path | None]:
    data_cfg = cfg.data
    analysis = cfg.analysis
    target = data_cfg.target_col
    exclude = data_cfg.exclude_cols + yearmonth_exclude

    with tqdm(total=8, desc="正在生成分析报告", unit="模块") as pbar:
        pbar.write(f"[INFO] 生成 {SHEET_SAMPLE} ...")
        sample_stats = monthly_y_distribution(df, target=target, yearmonth_col=yearmonth_col)
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_OVERVIEW} ...")
        overview = first_table(df, target=target, exclude_cols=exclude)
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_BINNING} ...")
        binning = binning_stats_chi(
            df,
            target=target,
            exclude_cols=exclude,
            min_samples=analysis.chi_min_samples,
        )
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_PSI} ...")
        psi = psi_against_base_toad(
            df,
            target=target,
            exclude_cols=exclude,
            date_col=data_cfg.date_col,
            yearmonth_col=yearmonth_col,
            base_month=analysis.psi_base_month,
        )
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_CORR} ...")
        corr = corr_matrix(df, target=target, exclude_cols=exclude)
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_MONTHLY} ...")
        monthly = variable_monthly_stats(
            df,
            target=target,
            yearmonth_col=yearmonth_col,
            base_month=analysis.psi_base_month,
            neg_corr_list=analysis.neg_corr_list,
            exclude_cols=exclude,
            date_col=data_cfg.date_col,
        )
        pbar.update(1)

        pbar.write(f"[INFO] 生成 {SHEET_KS} ...")
        ks = ks_bucket_all_vars(
            df,
            target=target,
            exclude_cols=exclude,
            bucket=analysis.ks_bucket,
            method=analysis.ks_method,
        )
        pbar.update(1)

        # 生成变量分布图
        if plots_dir is not None:
            pbar.write(f"[INFO] 生成 {SHEET_PLOTS} ...")
            plot_features_separately(
                df, plots_dir, exclude_cols=exclude, target_col=target, n_workers=12
            )
        pbar.update(1)

    tables = {
        SHEET_SAMPLE: sample_stats,
        SHEET_OVERVIEW: overview,
        SHEET_BINNING: binning,
        SHEET_PSI: psi,
        SHEET_CORR: corr,
        SHEET_MONTHLY: monthly,
        SHEET_KS: ks,
    }
    return tables, plots_dir
