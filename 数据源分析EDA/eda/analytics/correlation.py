"""变量相关性矩阵。"""

from __future__ import annotations

import pandas as pd

from eda.data import get_var_list


def corr_matrix(
    df: pd.DataFrame,
    target: str = "y_flag",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    features = get_var_list(df, exclude_cols=exclude_cols, target_col=target)

    # 识别全空列
    all_empty_cols = [col for col in features if df[col].isna().all()]

    corr = df[features].corr(method="pearson")

    # 全空列的相关性设为0
    for col in all_empty_cols:
        if col in corr.columns:
            corr.loc[corr.index, col] = 0.0
            corr.loc[col, :] = 0.0

    return corr.reset_index().rename(columns={"index": "变量名"})
