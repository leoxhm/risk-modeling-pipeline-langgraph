"""变量分布直方图导出。"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from eda.data import get_var_list


def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_single_feature(args: tuple) -> str:
    """并行绘制单个变量的分布图。"""
    var, data, folder, dpi = args
    _setup_font()
    plt.figure(figsize=(6.4, 4))
    sns.histplot(data, bins=20, kde=True, stat="density")
    plt.title(f"Distribution of {var}")
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.tight_layout()
    safe_name = var.replace(os.sep, "_")
    output_path = folder / f"{safe_name}.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return var


def plot_features_separately(
    df: pd.DataFrame,
    folder_name: str | Path,
    exclude_cols: list[str] | None = None,
    target_col: str = "y_flag",
    dpi: int = 80,
    n_workers: int | None = None,
) -> int:
    """并行生成变量分布图。"""
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target_col)
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    # 准备参数列表
    args_list = [(var, df[var].dropna(), folder, dpi) for var in features]

    # 并行执行
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_plot_single_feature, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成分布图"):
            future.result()  # 获取结果以捕获异常

    return len(features)
