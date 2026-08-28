from __future__ import annotations

import pandas as pd
import numpy as np
import toad
from toad.metrics import KS, PSI, AUC
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.cell.cell import Cell
from openpyxl.drawing.image import Image as XLImage
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter
import os
import shutil
import math
import tempfile
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve

# PMML导出支持
try:
    from sklearn2pmml import sklearn2pmml
    HAS_SKLEARN2PMML = True
except ImportError:
    HAS_SKLEARN2PMML = False
    print("⚠️ sklearn2pmml 未安装，PMML导出功能不可用。请运行: pip install sklearn2pmml")

# OLE对象插入支持（Windows专用）
try:
    import win32com.client as win32
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("⚠️ pywin32 未安装，OLE对象插入功能不可用。请运行: pip install pywin32")

PSI_GAP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ==================== Excel样式定义（遵循excel_writer.py规范） ====================
# 颜色定义
HEADER_FILL = PatternFill("solid", fgColor="E26B0A")
HEADER_FONT = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
BODY_FONT = Font(name="微软雅黑", size=10)
THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")

# 百分比列
PERCENT_COLS = {
    "缺失率", "完整率", "坏账率", "好账率", "好样本率", "样本占比",
    "坏样本占比", "好样本占比", "累计坏样本占比", "反向累计坏样本占比",
    "累计好样本占比", "反向累计好样本占比", "累计样本占比", "反向累计样本占比",
    "坏样本率", "好样本率", "占比", "占比%"
}

# 小标题格式：11号微软雅黑，加粗无边框，无自动换行，左对齐，黄色底
TITLE_FORMAT = PatternFill("solid", fgColor="FFFF00")
TITLE_FONT = Font(name="微软雅黑", size=11, bold=True, color="000000")

# 表格标题格式：11号微软雅黑，蓝底加粗带边框，自动换行，居中
TABLE_TITLE_FORMAT = PatternFill("solid", fgColor="5B9BD5")
TABLE_TITLE_FONT = Font(name="微软雅黑", size=11, bold=True, color="000000")

# 表格字符格式：11号微软雅黑，带边框，居中
TABLE_STR_FONT = Font(name="微软雅黑", size=11, bold=False, color="000000")

# 表格整数格式：11号微软雅黑，带边框，居中，整数格式
TABLE_INT_FORMAT = Font(name="微软雅黑", size=11, bold=False, color="000000")

# 表格小数格式：11号微软雅黑，带边框，居中，两位小数
TABLE_DECIMAL_FORMAT = Font(name="微软雅黑", size=11, bold=False, color="000000")

# 表格百分数格式：11号微软雅黑，带边框，居中，百分比格式
TABLE_PCT_FORMAT = Font(name="微软雅黑", size=11, bold=False, color="000000")


# ==================== 辅助函数（遵循excel_writer.py规范） ====================

def _excel_str_width(s: str) -> float:
    """计算字符串在Excel中的显示宽度（中文按2倍字符宽度计算）。"""
    if not s:
        return 0
    width = 0
    for char in str(s):
        if ord(char) > 255:  # 中文字符
            width += 2
        else:
            width += 1
    return width

def cal_col_width(ws, current_row):
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        header = ws.cell(current_row, col_idx).value
        header_str = str(header or "")
        max_width = _excel_str_width(header_str)
        if max_width <= 8:
            new_width = 11
        else:
            new_width = max_width + 3
        current_width = ws.column_dimensions[col_letter].width
        if current_width:
            new_width = max(new_width, current_width)
        ws.column_dimensions[col_letter].width = new_width

def _number_format(col_name: str) -> str | None:
    """根据列名返回对应的数字格式"""
    # 百分比列
    if col_name in PERCENT_COLS or any(col_name.endswith(s) for s in ("率", "占比", "比")):
        return "0.00%"
    
    # 4位小数列
    if col_name in {"IV值", "IV", "KS值", "KS", "平均PSI", "最大PSI", "PSI值", "AUC值", "AUC", "WOE值", "分箱IV值", "累计IV值", "累计KS值",
                    "训练集IV", "测试集IV", "验证集IV", "全数据集IV", "基尼系数", "信息熵", "PSI(训练->测试)", "PSI(训练->验证)",
                    "平均值", "协方差", "最小值", "最大值", "1%", "10%", "25%", "50%", "75%", "90%", "99%", "缺失率"}:
        return "0.0000"
    
    # 整数列
    if col_name in {"唯一值数", "唯一值", "好样本", "坏样本", "总样本", "好样本数", "坏样本数", "总样本数", "样本总量", "样本数", "总样本数", "总数"}:
        return "#,##0"
    
    # 2位小数列
    if col_name in {"标准差", "1%分位", "10%分位", "50%分位", "75%分位", "90%分位", "99%分位", "Lift值", "累计Lift值", "Lift前10%", "好坏比", "分箱", "LIFT"}:
        return "#,##0.00"
    
    # 文本列
    if col_name in ("变量名", "分箱区间", "稳定性评估", "年月", "数据集", "类型", "PSI指标"):
        return "@"
    
    return "#,##0.00"


def _apply_global_style(ws, df=None, freeze_panes=False, autofilter=False):
    """应用全局样式（遵循excel_writer.py规范）"""
    # 找到年月列索引
    yearmonth_col_idx = None
    for col_idx in range(1, ws.max_column + 1):
        if ws.cell(1, col_idx).value == "年月":
            yearmonth_col_idx = col_idx
            break

    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.border = THIN_BORDER
            if cell.row == 1:
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = CENTER
            else:
                # 年月列强制转换为文本
                if cell.column == yearmonth_col_idx:
                    cell.value = str(cell.value) if cell.value is not None else ""
                    cell.number_format = "@"
                    cell.alignment = CENTER
                    cell.font = BODY_FONT
                    continue

                cell.font = BODY_FONT
                col_name = ws.cell(1, cell.column).value
                if col_name in ("变量名", "分箱区间"):
                    cell.alignment = LEFT
                else:
                    cell.alignment = CENTER
                fmt = _number_format(str(col_name) if col_name else "")
                if fmt and isinstance(cell.value, (int, float)):
                    cell.number_format = fmt

    # 设置行高：第一行30，其他行20
    ws.row_dimensions[1].height = 30
    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 20

    # 设置列宽：第一列22，其他列根据header字符数计算
    ws.column_dimensions["A"].width = 22
    for col_idx in range(2, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        header = ws.cell(1, col_idx).value
        header_str = str(header or "")
        width = _excel_str_width(header_str)
        # 基础宽度11，小于等于4字符保持11
        if width <= 8:
            ws.column_dimensions[col_letter].width = 11
        else:
            ws.column_dimensions[col_letter].width = width + 3

    if freeze_panes and ws.max_row > 1:
        ws.freeze_panes = "A2"
    if autofilter and ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions


######################
## 报告1：模型综合效果
######################

def model_report_01(model_data, model_score, y_flag, data_flag="samp_type", ym_flag=None, n_bins=10, ascending=False):
    result = {}
    type_list = sorted(
        model_data[data_flag].dropna().unique().tolist(),
        key=lambda x: (
            {v: i for i, v in enumerate(['train', 'test', 'vldt'])}.get(str(x).lower(), 3),
            x
        )
    )
    data_by_type = {t: model_data[model_data[data_flag] == t] for t in type_list}
    # ---------- 1. 样本分布 ----------
    all_rows = []
    for t in type_list:
        sub = data_by_type[t]
        bad_cnt = int(sub[y_flag].sum())
        obs_cnt = len(sub)
        row = {
            '数据集': t,
            '样本数': obs_cnt,
            '坏样本数': bad_cnt,
            '好样本数': obs_cnt - bad_cnt,
            '坏样本率': round(bad_cnt / obs_cnt, 4) if obs_cnt > 0 else None,
        }
        if len(sub) == 0:
            row['KS'] = None
            row['AUC'] = None
        else:
            row['KS'] = round(KS(sub[model_score], sub[y_flag]), 4)
            row['AUC'] = round(AUC(sub[model_score], sub[y_flag]), 4)
        all_rows.append(row)
    result['模型效果汇总'] = pd.DataFrame(all_rows)

    # ---------- 2. 模型效果_PSI ----------
    train_pred = data_by_type['train'][model_score] if 'train' in data_by_type else pd.Series(dtype=float)
    test_pred = data_by_type['test'][model_score] if 'test' in data_by_type else pd.Series(dtype=float)
    vldt_pred = data_by_type['vldt'][model_score] if 'vldt' in data_by_type else pd.Series(dtype=float)

    psi_rows = []
    if len(train_pred) > 0 and len(test_pred) > 0:
        psi_rows.append({
            '数据集PSI指标': 'train -> test',
            'PSI值': round(PSI(train_pred, test_pred, PSI_GAP), 4)
        })
    if len(train_pred) > 0 and len(vldt_pred) > 0:
        psi_rows.append({
            '数据集PSI指标': 'train -> vldt',
            'PSI值': round(PSI(train_pred, vldt_pred, PSI_GAP), 4)
        })
    if len(train_pred) > 0 and len(test_pred) > 0 and len(vldt_pred) > 0:
        psi_rows.append({
            '数据集PSI指标': 'train_test -> vldt',
            'PSI值': round(PSI(pd.concat([train_pred, test_pred]), vldt_pred, PSI_GAP), 4)
        })

    result['模型效果_PSI'] = pd.DataFrame(psi_rows) if psi_rows else pd.DataFrame(columns=['数据集PSI指标', 'PSI值'])

    # ---------- 3. 各月分布（外层数据集 → 内层月份）----------
    if ym_flag and ym_flag in model_data.columns:
        ym_list = sorted(model_data[ym_flag].dropna().unique())
        ym_rows = []
        
        for t in type_list:
            for ym in ym_list:
                ym_sub = model_data[model_data[ym_flag] == ym]
                sub_t = ym_sub[ym_sub[data_flag] == t]
                if len(sub_t) == 0:
                    continue
                bad_cnt = int(sub_t[y_flag].sum())
                obs_cnt = len(sub_t)
                ym_rows.append({
                    '数据集': t,
                    '月份': ym,
                    '样本数': obs_cnt,
                    '坏样本数': bad_cnt,
                    '好样本数': obs_cnt - bad_cnt,
                    '坏样本率': round(bad_cnt / obs_cnt, 4) if obs_cnt > 0 else None
                })
        
        ym_df = pd.DataFrame(ym_rows)
        
        if len(ym_df) > 0:
            # 计算总坏样本数和总样本数（用于计算各月分布的比例）
            total_bad = ym_df['坏样本数'].sum()
            total_obs = ym_df['样本数'].sum()
            # 在各月分布中添加坏样本占总坏比和占总样本比
            ym_df['坏占总坏比'] = round(ym_df['坏样本数'] / total_bad, 4) if total_bad > 0 else 0
            ym_df['样本占比'] = round(ym_df['样本数'] / total_obs, 4) if total_obs > 0 else 0
            result['各月分布'] = ym_df
            
            summary_df = ym_df.groupby('数据集').agg(
                总样本数=('样本数', 'sum'),
                总坏样本数=('坏样本数', 'sum'),
                总好样本数=('好样本数', 'sum'),
                平均坏样本率=('坏样本率', 'mean'),
                覆盖月份数=('月份', 'nunique')
            ).reset_index()
            summary_df['数据集'] = pd.Categorical(summary_df['数据集'], categories=type_list, ordered=True)
            summary_df = summary_df.sort_values('数据集').reset_index(drop=True)
            # 计算汇总的坏样本占总坏比和占总样本比
            total_bad_sum = summary_df['总坏样本数'].sum()
            total_obs_sum = summary_df['总样本数'].sum()
            summary_df['平均坏样本率'] = round(summary_df['平均坏样本率'],4)
            summary_df['坏占总坏比'] = round(summary_df['总坏样本数'] / total_bad_sum, 4) if total_bad_sum > 0 else 0
            summary_df['样本占比'] = round(summary_df['总样本数'] / total_obs_sum, 4) if total_obs_sum > 0 else 0
            result['各月分布_汇总'] = summary_df
        else:
            result['各月分布'] = None
            result['各月分布_汇总'] = None
    else:
        result['各月分布'] = None
        result['各月分布_汇总'] = None

    # ---------- 4. LIFT 分箱 ----------
    df_clean = model_data[[model_score, y_flag]].dropna()
    total_obs = len(df_clean)
    total_bad = int(df_clean[y_flag].sum())
    total_good = total_obs - total_bad

    df_sorted = df_clean.sort_values(by=model_score, ascending=ascending).reset_index(drop=True)
    df_sorted['bin'] = pd.qcut(df_sorted.index, q=n_bins, labels=False) + 1

    lift_df = df_sorted.groupby('bin').agg(
        bad_cnt=(y_flag, 'sum'),
        obs_cnt=(y_flag, 'count')
    ).reset_index()
    lift_df['good_cnt'] = lift_df['obs_cnt'] - lift_df['bad_cnt']
    lift_df['bad_pct'] = (lift_df['bad_cnt'] / total_bad * 100).round(1) if total_bad > 0 else 0
    lift_df['good_pct'] = (lift_df['good_cnt'] / total_good * 100).round(1) if total_good > 0 else 0
    lift_df['obs_pct'] = (lift_df['obs_cnt'] / total_obs * 100).round(1)
    lift_df['lift'] = (lift_df['bad_pct'] / lift_df['obs_pct']).round(2) if lift_df['obs_pct'].sum() > 0 else 0
    lift_df.columns = ['分箱', '坏样本数', '样本数', '好样本数', '坏样本占比%', '好样本占比%', '样本占比%', 'LIFT']
    result['LIFT分箱'] = lift_df

    return result


######################
## 报告2：模型变量KS分析（从base_code.py提取）
######################

def model_vars_csv(woe, slc_cols, train_model_var, test_model_var, vldt_model_var, train_y, test_y, vldt_y):
    """
    生成模型变量KS分析数据
    :param woe: WOE映射字典
    :param slc_cols: 选中的变量列表
    :param train_model_var: 训练集模型变量
    :param test_model_var: 测试集模型变量
    :param vldt_model_var: 验证集模型变量
    :param train_y: 训练集标签
    :param test_y: 测试集标签
    :param vldt_y: 验证集标签
    :return: KS分析结果DataFrame
    """
    ks_res = pd.DataFrame()
    for col in slc_cols:
        temp = dict(zip(woe[col].values(), woe[col].keys()))
        ks_temp_res = toad.KS_bucket(
            pd.concat([train_model_var[col], test_model_var[col], vldt_model_var[col]]),
            pd.concat([train_y, test_y, vldt_y]),
            bucket=sorted(list(set(train_model_var[col])))
        )
        ks_temp_res['bin'] = ks_temp_res['min'].apply(lambda x: temp[x])
        ks_temp_res['var'] = col
        ks_temp_res = ks_temp_res[[
            'var', 'bin', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 
            'odds', 'bad_prop', 'good_prop', 'total_prop', 
            'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks'
        ]]
        ks_temp_res.sort_values(by='bin')
        ks_res = pd.concat([ks_res, ks_temp_res])
    
    ks_res.columns = [
        "变量名", "分箱", "坏样本量", "好样本量", "总样本量", "坏样本率", "好样本率", "赔率", 
        "坏样本占全部坏样本比例", "好样本占全部好样本比例", "组内样本量占比", 
        "累计坏样本率", "累计好样本率", "累计样本率", "单变量ks"
    ]
    return ks_res


######################
## 报告3：入模变量基本信息
######################

def model_report_03(model_data, model, y_flag, ym_flag=None, samp_type='samp_type'):
    model_vars = model.booster_.feature_name()

    iv_result = toad.quality(
        model_data[model_vars + [y_flag]],
        target=y_flag,
        iv_only=True,
        cpu_cores=16
    )
    
    iv_result = iv_result.reset_index()
    iv_result = iv_result.iloc[:, [0, 1]]
    iv_result.columns = ['变量名', 'IV']
    iv_result['IV'] = iv_result['IV'].round(4)

    var_imp = pd.DataFrame({
        "变量名": model_vars,
        "重要性": model.feature_importances_
    }).sort_values(by="重要性", ascending=False)

    result_df = var_imp.merge(iv_result, how="left", on='变量名').copy()
    
    # 在IV后增加：总数、缺失率、唯一值数量
    basic_stats_dict = {
        '变量名': [],
        '总数': [],
        '缺失率': [],
        '唯一值数': []
    }
    
    for col in model_vars:
        basic_stats_dict['变量名'].append(col)
        col_data = model_data[col]
        total_count = len(col_data)
        missing_count = col_data.isna().sum()
        unique_count = col_data.nunique()
        
        basic_stats_dict['总数'].append(total_count)
        basic_stats_dict['缺失率'].append(round(missing_count / total_count, 4) if total_count > 0 else 0)
        basic_stats_dict['唯一值数'].append(unique_count)
    
    basic_stats_df = pd.DataFrame(basic_stats_dict)
    result_df = result_df.merge(basic_stats_df, how='left', on='变量名')
    
    unique_samp_types = model_data[samp_type].unique()
    train_keys = [k for k in unique_samp_types if 'train' in str(k).lower()]
    test_keys = [k for k in unique_samp_types if 'test' in str(k).lower()]
    vldt_keys = [k for k in unique_samp_types if 'vldt' in str(k).lower() or 'valid' in str(k).lower() or '外推' in str(k)]
    
    train_data = model_data[model_data[samp_type].isin(train_keys)] if train_keys else pd.DataFrame()
    test_data = model_data[model_data[samp_type].isin(test_keys)] if test_keys else pd.DataFrame()
    vldt_data = model_data[model_data[samp_type].isin(vldt_keys)] if vldt_keys else pd.DataFrame()
    
    # 使用字典一次性构建DataFrame
    analyse_dict = {
        '变量名': [],
        '训练集IV': [],
        '测试集IV': [],
        '验证集IV': [],
        '基尼系数': [],
        '信息熵': [],
        'PSI(训练->测试)': [],
        'PSI(训练->验证)': []
    }
    
    for col in model_vars:
        analyse_dict['变量名'].append(col)
        
        # 处理训练集IV
        try:
            analyse_dict['训练集IV'].append(round(toad.stats.IV(train_data[col], train_data[y_flag]), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['训练集IV'].append(None)
        
        # 处理测试集IV
        try:
            analyse_dict['测试集IV'].append(round(toad.stats.IV(test_data[col], test_data[y_flag]), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['测试集IV'].append(None)
        
        # 处理外推集IV
        try:
            analyse_dict['验证集IV'].append(round(toad.stats.IV(vldt_data[col], vldt_data[y_flag]), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['验证集IV'].append(None)
        
        # 处理基尼系数
        try:
            analyse_dict['基尼系数'].append(round(toad.stats.gini(model_data[col]), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['基尼系数'].append(None)
        
        # 处理信息熵
        try:
            analyse_dict['信息熵'].append(round(toad.stats.entropy(model_data[col]), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['信息熵'].append(None)
        
        # 处理PSI（训练->测试）
        try:
            analyse_dict['PSI(训练->测试)'].append(round(toad.metrics.PSI(train_data[col], test_data[col], PSI_GAP), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['PSI(训练->测试)'].append(None)
        
        # 处理PSI（训练->验证）
        try:
            analyse_dict['PSI(训练->验证)'].append(round(toad.metrics.PSI(train_data[col], vldt_data[col], PSI_GAP), 4))
        except (ZeroDivisionError, ValueError):
            analyse_dict['PSI(训练->验证)'].append(None)
    
    analyse_df = pd.DataFrame(analyse_dict)
    
    # 将 analyse_df 的字段合并到 result_df
    result_df = result_df.merge(analyse_df, how='left', on='变量名')

    if ym_flag is None or ym_flag not in model_data.columns:
        return result_df

    # 在月份PSI前增加总体统计信息：平均值、协方差、最小值、最大值、1%、10%、25%、50%、75%、90%、99%
    stats_dict = {
        '变量名': [],
        '平均值': [],
        '协方差': [],
        '最小值': [],
        '最大值': [],
        '1%': [],
        '10%': [],
        '25%': [],
        '50%': [],
        '75%': [],
        '90%': [],
        '99%': []
    }
    
    for col in model_vars:
        stats_dict['变量名'].append(col)
        col_data = model_data[col].dropna()
        if len(col_data) > 0:
            stats_dict['平均值'].append(round(col_data.mean(), 4))
            stats_dict['协方差'].append(round(col_data.var(), 4))
            stats_dict['最小值'].append(round(col_data.min(), 4))
            stats_dict['最大值'].append(round(col_data.max(), 4))
            stats_dict['1%'].append(round(col_data.quantile(0.01), 4))
            stats_dict['10%'].append(round(col_data.quantile(0.10), 4))
            stats_dict['25%'].append(round(col_data.quantile(0.25), 4))
            stats_dict['50%'].append(round(col_data.quantile(0.50), 4))
            stats_dict['75%'].append(round(col_data.quantile(0.75), 4))
            stats_dict['90%'].append(round(col_data.quantile(0.90), 4))
            stats_dict['99%'].append(round(col_data.quantile(0.99), 4))
        else:
            stats_dict['平均值'].append(None)
            stats_dict['协方差'].append(None)
            stats_dict['最小值'].append(None)
            stats_dict['最大值'].append(None)
            stats_dict['1%'].append(None)
            stats_dict['10%'].append(None)
            stats_dict['25%'].append(None)
            stats_dict['50%'].append(None)
            stats_dict['75%'].append(None)
            stats_dict['90%'].append(None)
            stats_dict['99%'].append(None)
    
    stats_df = pd.DataFrame(stats_dict)
    result_df = result_df.merge(stats_df, how='left', on='变量名')

    # 使用首月作为基准计算分月PSI
    yearmonth_list = sorted(model_data[ym_flag].unique())
    ben_ym = yearmonth_list[0]
    
    # 提取基准数据（首月）
    base_data = model_data[model_data[ym_flag] == ben_ym]
    
    # 使用字典收集所有月份的 PSI 数据
    psi_data = {}
    for ym in yearmonth_list:
        current_subset = model_data[model_data[ym_flag] == ym]
        psi_list = []
        for col in model_vars:
            if ym == ben_ym:
                # 首月PSI为0
                psi_list.append(0.0)
            else:
                current_col_data = current_subset[col]
                base_col_data = base_data[col]
                if len(current_col_data) == 0 or len(base_col_data) == 0:
                    psi_list.append(None)
                else:
                    # 使用toad.metrics.PSI，基准数据在前，当前数据在后
                    psi_list.append(round(PSI(base_col_data, current_col_data, PSI_GAP), 4))
        psi_data[ym] = psi_list
    
    # 一次性合并所有月份的 PSI 数据
    psi_df = pd.DataFrame(psi_data, index=result_df.index)
    result_df = pd.concat([result_df, psi_df], axis=1)

    return result_df


######################
## 报告4：变量分箱 - 是否单调（含完整分箱统计用于画图）
######################

def model_report_04(model_data, model, y_flag, n_bins=10):
    model_vars = model.booster_.feature_name()
    records = []
    detail_records = []

    for col in model_vars:
        df_sub = model_data[[col, y_flag]].dropna()
        if len(df_sub) == 0:
            continue

        bad_rates = []
        bin_labels = []
        bin_stats_list = []

        try:
            c = toad.transform.Combiner()
            c.fit(df_sub[[col, y_flag]], y=y_flag, method='chi', n_bins=min(n_bins, len(df_sub)//2))
            bins = c.export()[col]
            df_sub['bin'] = pd.cut(df_sub[col], bins=bins, include_lowest=True)
            bin_stats = df_sub.groupby('bin', observed=False).agg(
                count=(y_flag, 'count'),
                bad_count=(y_flag, 'sum')
            ).reset_index()
            bin_stats['good_count'] = bin_stats['count'] - bin_stats['bad_count']
            bin_stats['bad_rate'] = (bin_stats['bad_count'] / bin_stats['count']).round(4)
            total_cnt = bin_stats['count'].sum()
            bin_stats['total_pct'] = (bin_stats['count'] / total_cnt).round(4)
            bin_stats['good_pct'] = (bin_stats['good_count'] / total_cnt).round(4)
            bin_stats['bad_pct'] = (bin_stats['bad_count'] / total_cnt).round(4)
            bad_rates = bin_stats['bad_rate'].dropna().tolist()
            bin_labels = [f"箱{i+1}" for i in range(len(bad_rates))]

        except Exception:
            try:
                df_sub['bin'] = pd.qcut(
                    df_sub[col].rank(method='first'), 
                    q=n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
                bin_stats = df_sub.groupby('bin', observed=False).agg(
                    count=(y_flag, 'count'),
                    bad_count=(y_flag, 'sum')
                ).reset_index()
                bin_stats['good_count'] = bin_stats['count'] - bin_stats['bad_count']
                bin_stats['bad_rate'] = (bin_stats['bad_count'] / bin_stats['count']).round(4)
                total_cnt = bin_stats['count'].sum()
                bin_stats['total_pct'] = (bin_stats['count'] / total_cnt).round(4)
                bin_stats['good_pct'] = (bin_stats['good_count'] / total_cnt).round(4)
                bin_stats['bad_pct'] = (bin_stats['bad_count'] / total_cnt).round(4)
                bad_rates = bin_stats['bad_rate'].dropna().tolist()
                bin_labels = [f"箱{i+1}" for i in range(len(bad_rates))]
            except Exception as e2:
                records.append({
                    '变量名': col,
                    '是否单调': '异常',
                    '单调趋势': str(e2)[:50],
                    '分箱数': 0,
                    '最小bad_rate': None,
                    '最大bad_rate': None
                })
                continue

        for idx, row in bin_stats.iterrows():
            # 使用 Interval 对象的字符串表示作为分箱标签（如 [0, 100)）
            bin_label = str(row['bin']) if 'bin' in row else (bin_labels[idx] if idx < len(bin_labels) else f"箱{idx+1}")
            detail_records.append({
                '变量名': col,
                '分箱序号': idx + 1,
                '分箱标签': bin_label,
                'count': int(row['count']),
                'bad_count': int(row['bad_count']),
                'good_count': int(row['good_count']),
                'bad_rate': round(row['bad_rate'], 4),
                'total_pct': round(row['total_pct'], 4),
                'good_pct': round(row['good_pct'], 4),
                'bad_pct': round(row['bad_pct'], 4)
            })

        is_mono, trend = False, '非单调'
        if len(bad_rates) >= 2:
            inc = all(bad_rates[i] <= bad_rates[i+1] + 1e-9 for i in range(len(bad_rates)-1))
            dec = all(bad_rates[i] >= bad_rates[i+1] - 1e-9 for i in range(len(bad_rates)-1))
            if inc:
                is_mono, trend = True, '单调递增'
            elif dec:
                is_mono, trend = True, '单调递减'

        records.append({
            '变量名': col,
            '是否单调': '是' if is_mono else '否',
            '单调趋势': trend,
            '分箱数': len(bad_rates),
            '最小bad_rate': round(min(bad_rates), 4) if bad_rates else None,
            '最大bad_rate': round(max(bad_rates), 4) if bad_rates else None
        })

    return {
        '汇总': pd.DataFrame(records),
        '分箱明细': pd.DataFrame(detail_records)
    }


######################
## 报告5：跨周期稳定性分析（从base_code.py提取）
######################

def crss_prd_vld(model, df_x, df_y, loc_var, baseline, stp_slc, tgt_col, output_dir, out_flag=False, bucket_num=10):
    """
    跨周期验证
    :param model: 模型对象
    :param df_x: 特征数据
    :param df_y: 标签数据
    :param loc_var: 分组变量（如年月）
    :param baseline: 基准预测值
    :param stp_slc: 选中的变量列表
    :param tgt_col: 目标列名
    :param out_flag: 是否输出文件
    :param bucket_num: 分箱数
    :return: 跨周期验证结果DataFrame
    """
    ym_ks = []
    ym_auc = []
    ym_psi = []
    ym_list = np.unique(df_x[loc_var])

    for ym in ym_list:
        x = df_x[df_x[loc_var] == ym][stp_slc]
        y = df_y[df_y[loc_var] == ym][tgt_col]

        y_pred = model.predict_proba(x)[:, 1]
        
        ym_auc.append(AUC(y_pred, y))
        ym_ks.append(KS(y_pred, y))
        ym_psi.append(PSI(baseline, y_pred, PSI_GAP))
        
        if out_flag:
            toad.KS_bucket(y_pred, list(y), bucket=bucket_num)[[
                'min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate',
                'odds', 'bad_prop', 'good_prop', 'total_prop',
                'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks'
            ]].to_csv(f"{output_dir}/{ym}_ks.csv", index=None, sep='\t')
            
    return pd.DataFrame({'ym_ks': ym_ks, 'ym_auc': ym_auc, 'ym_psi': ym_psi}, index=ym_list)


######################
## 报告6：模型变量相关性分析（从base_code.py提取）
######################

def model_var_corr(train_model_var, output_dir="output"):
    """
    模型变量相关性分析
    :param train_model_var: 训练集模型变量
    :param output_dir: 输出目录
    :return: 相关性分析结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算相关性矩阵
    corr_df = train_model_var.corr()
    
    # 生成相关性矩阵表
    model_var_corr = corr_df.reset_index()
    model_var_corr.rename(columns={"index": "model_var"}, inplace=True)
    model_var_corr = model_var_corr.apply(lambda x: x.round(2) if x.dtype in ['float64', 'float32'] else x)
    model_var_corr.to_csv(f"{output_dir}/model_var_corr.csv", encoding="utf8", sep=",", index=False)
    
    # 生成相关性Top10表（序号、变量1、变量2、相关系数）
    corr_pairs = []
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):  # 只取上三角，避免重复
            corr_pairs.append({
                '变量1': cols[i],
                '变量2': cols[j],
                '相关系数': round(corr_df.iloc[i, j], 4)
            })
    
    corr_top10_df = pd.DataFrame(corr_pairs)
    # 按相关系数绝对值排序，取Top10
    corr_top10_df['abs_corr'] = corr_top10_df['相关系数'].abs()
    corr_top10_df = corr_top10_df.sort_values('abs_corr', ascending=False).head(10)
    corr_top10_df = corr_top10_df.drop('abs_corr', axis=1).reset_index(drop=True)
    corr_top10_df.insert(0, '序号', range(1, len(corr_top10_df) + 1))
    corr_top10_df.to_csv(f"{output_dir}/model_var_corr_top10.csv", encoding="utf8", sep=",", index=False)
    
    return {
        '相关性矩阵': model_var_corr,
        '相关性Top10': corr_top10_df
    }


######################
## 辅助函数：分布计算（从base_code.py提取）
######################

def dist_calc(df_x_bin_temp, df_y, col):
    """
    判断违约率是否单调并计算分布
    
    Arguments:
    df_x_bin_temp: 离散化后的DataFrame
    df_y: 标签数据
    col: 分析变量
    
    Returns:
    iv: iv值
    mono_flag: 0-不单调; 1-单调递增; -1-单调递减
    df_temp: DataFrame,最终分布
    """
    df_y_ = pd.DataFrame(df_y)
    df_y_.columns = ['y']
    df_xy_bin_temp = pd.concat([df_x_bin_temp, df_y_], axis=1)
    df_temp = df_xy_bin_temp.groupby(col)['y'].agg(
        ['count', 'sum']).reset_index().rename(columns={'sum': 'bad'})
    df_temp = df_temp.assign(
        good=df_temp['count'] - df_temp['bad'],
        d_all=df_temp['count'] / sum(df_temp['count']),
        d_bad=df_temp['bad'] / sum(df_temp['bad'])).assign(
            d_good=lambda x: x['good'] / sum(x['good']),
            p_bad=lambda x: x['bad'] / sum(x['count']),
            p_good=lambda x: x['good'] / sum(x['count']),
            bad_rate=lambda x: x['bad'] / x['count']).assign(
                bin_iv=lambda x:
                (x['d_bad'] - x['d_good']) * np.log(x['d_bad'] / x['d_good']),
                bin_woe=lambda x: np.log(x['d_bad'] / x['d_good']))
    
    # 判断单调性
    bad_rate = df_temp['bad_rate'].values
    inc = np.all(np.diff(bad_rate) >= 0)
    dec = np.all(np.diff(bad_rate) <= 0)
    
    if inc:
        mono_flag = 1
    elif dec:
        mono_flag = -1
    else:
        mono_flag = 0
    
    iv = df_temp['bin_iv'].sum()
    
    return iv, mono_flag, df_temp


def cross_calc(df_bin, by_col, col, tgt_col):
    """
    判断分组后每组的违约率趋势是否相同
    
    Arguments:
    df_bin: 离散化后的DataFrame
    by_col: 分组变量
    col: 分析变量
    tgt_col: 因变量   
    """
    df_cross = df_bin.groupby([by_col, col])[tgt_col].agg(
        ['count', 'sum']).reset_index().rename(columns={
            'sum': 'bad'
        }).assign(bad_rate=lambda x: x['bad'] / x['count']).drop(
            columns=['count', 'bad']).pivot(index=by_col, columns=col, values='bad_rate')

    if re.search('nan', str(df_cross.columns[-1])):
        temp = df_cross.iloc[:, :-1].values
    else:
        temp = df_cross.values

    a = np.all(np.diff(temp, axis=1) <= 0, axis=0)
    b = np.all(np.diff(temp, axis=1) >= 0, axis=0)

    return np.all(a | b), df_cross


######################
## 缺失图表：单变量样本分布及违约趋势图（从base_code.py提取）
######################

def plot_bin(df_bin, df_y, col, title=None, show_iv=True, save_path=None):
    """
    单变量样本分布及违约趋势图
    
    Params
    ------
    df_bin: 离散化后的DataFrame
    df_y: 标签数据
    col: 分析变量
    title: 标题前置文本
    show_iv: 是否显示iv值
    save_path: 保存路径（None表示不保存）
    
    Returns
    ------
    matplotlib fig object
    """
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    temp = dist_calc(df_bin, df_y, col)
    iv = temp[0]
    dist_binx = temp[2]

    x_list = dist_binx[col]
    bad_rate = dist_binx['bad_rate']
    bin_dist = dist_binx['d_all']
    bin_count = dist_binx['count']
    good_dist = dist_binx['p_good']
    bad_dist = dist_binx['p_bad']

    y_right_max = np.ceil(bad_rate.max() * 10)
    if y_right_max % 2 == 1:
        y_right_max = y_right_max + 1
    if y_right_max - bad_rate.max() * 10 <= 0.3:
        y_right_max = y_right_max + 2

    y_right_max = y_right_max / 10
    if y_right_max > 1 or y_right_max <= 0 or np.isnan(y_right_max) or y_right_max is None:
        y_right_max = 1
    y_left_max = np.ceil(bin_dist.max() * 10) / 10
    if y_left_max > 1 or y_left_max <= 0 or np.isnan(y_left_max) or y_left_max is None:
        y_left_max = 1

    title_string = f"{col}  (iv:{round(iv, 4)})" if show_iv else col
    title_string = f"{title}-{title_string}" if title is not None else title_string

    ind = np.arange(len(bad_rate))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    
    p1 = ax1.bar(ind, good_dist, width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, bad_dist, width, bottom=good_dist, color=(246/254, 115/254, 109/254))
    
    for i in ind:
        ax1.text(i, bin_dist[i] * 1.02, f"{round(bin_dist[i]*100, 1)}%, {bin_count[i]}", ha='center')
    
    ax2.plot(ind, bad_rate, marker='o', color='blue')
    for i in ind:
        ax2.text(i, bad_rate[i] * 1.02, f"{round(bad_rate[i]*100, 1)}%", color='blue', ha='center')

    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')

    plt.xticks(ind, x_list)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        return save_path
    
    plt.close(fig)
    return fig


######################
## 缺失图表：分组违约率趋势图（从base_code.py提取）
######################

def plot_badrate(df_bin, by_col, col, tgt_col, save_path=None):
    """
    分组违约率趋势图
    
    Arguments:
    df_bin: 离散化后的DataFrame
    by_col: 分组变量
    col: 分析变量
    tgt_col: 因变量
    save_path: 保存路径（None表示不保存）
    
    Returns:
    matplotlib ax object or save path
    """
    import matplotlib.style as psl
    psl.use('seaborn-ticks')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        temp = cross_calc(df_bin, by_col, col, tgt_col)[1]
        fig, ax = plt.subplots(figsize=(4.5, 2.8), dpi=300)
        temp.plot(
            ax=ax,
            title=col,
            style=['*-', '^-', '.-', 'p-', 'o-', '*--', '^--', '-.', ':', '.', '-'])
        ax.set_ylabel('Bad rate')
        ax.legend(loc='best')
        ax.xaxis.label.set_visible(False)
        
        if save_path:
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
            return save_path
        
        plt.close()
        return ax

######################
## 报告8：入模变量概况（从base_code.py提取）
######################

def model_vars_analyse(ds_all, tgt_col, model, samp_type='samp_type', yearmonth='yearmonth'):
    """
    入模变量概况分析（IV、Gini、Entropy、PSI）（性能优化版）
    :param ds_all: 完整数据集
    :param tgt_col: 目标列名
    :param model: 模型对象
    :param samp_type: 样本类型列名
    :param yearmonth: 年月列名
    :return: 入模变量概况DataFrame
    """
    var_list = model.booster_.feature_name()
    
    # 自动识别样本类型格式 - 支持 'train'/'01.train' 等多种格式
    unique_samp_types = ds_all[samp_type].unique()
    train_keys = [k for k in unique_samp_types if 'train' in str(k).lower()]
    test_keys = [k for k in unique_samp_types if 'test' in str(k).lower()]
    vldt_keys = [k for k in unique_samp_types if 'vldt' in str(k).lower() or 'valid' in str(k).lower() or '外推' in str(k)]
    
    train_data = ds_all[ds_all[samp_type].isin(train_keys)] if train_keys else pd.DataFrame()
    test_data = ds_all[ds_all[samp_type].isin(test_keys)] if test_keys else pd.DataFrame()
    vldt_data = ds_all[ds_all[samp_type].isin(vldt_keys)] if vldt_keys else pd.DataFrame()
    
    # 使用字典一次性构建DataFrame，避免逐列赋值
    result_dict = {
        '变量名': [],
        '训练集IV': [],
        '测试集IV': [],
        '外推集IV': [],
        '全数据集IV': [],
        '基尼系数': [],
        '信息熵': [],
        'PSI(训练->测试)': [],
        'PSI(训练->外放)': []
    }
    
    for col in var_list:
        result_dict['变量名'].append(col)
        
        # 处理训练集IV（避免除零错误）
        try:
            result_dict['训练集IV'].append(toad.stats.IV(train_data[col], train_data[tgt_col]))
        except (ZeroDivisionError, ValueError):
            result_dict['训练集IV'].append(None)
        
        # 处理测试集IV
        try:
            result_dict['测试集IV'].append(toad.stats.IV(test_data[col], test_data[tgt_col]))
        except (ZeroDivisionError, ValueError):
            result_dict['测试集IV'].append(None)
        
        # 处理外推集IV
        try:
            result_dict['外推集IV'].append(toad.stats.IV(vldt_data[col], vldt_data[tgt_col]))
        except (ZeroDivisionError, ValueError):
            result_dict['外推集IV'].append(None)
        
        # 处理全数据集IV
        try:
            result_dict['全数据集IV'].append(toad.stats.IV(ds_all[col], ds_all[tgt_col]))
        except (ZeroDivisionError, ValueError):
            result_dict['全数据集IV'].append(None)
        
        # 处理基尼系数
        try:
            result_dict['基尼系数'].append(toad.stats.gini(ds_all[col]))
        except (ZeroDivisionError, ValueError):
            result_dict['基尼系数'].append(None)
        
        # 处理信息熵
        try:
            result_dict['信息熵'].append(toad.stats.entropy(ds_all[col]))
        except (ZeroDivisionError, ValueError):
            result_dict['信息熵'].append(None)
        
        # 处理PSI（训练->测试）- 基准在前，当前在后
        try:
            result_dict['PSI(训练->测试)'].append(toad.metrics.PSI(train_data[col], test_data[col], PSI_GAP))
        except (ZeroDivisionError, ValueError):
            result_dict['PSI(训练->测试)'].append(None)
        
        # 处理PSI（训练->外放）- 基准在前，当前在后
        try:
            result_dict['PSI(训练->外放)'].append(toad.metrics.PSI(train_data[col], vldt_data[col], PSI_GAP))
        except (ZeroDivisionError, ValueError):
            result_dict['PSI(训练->外放)'].append(None)
    
    return pd.DataFrame(result_dict)


######################
## 报告9：模型结果汇总（含图表）（从base_code.py提取）
######################

def plot_ks(fpr, tpr, title, save_fig=False, output_dir="output"):
    """
    绘制KS曲线（内存优化版）
    :param fpr: 假正率
    :param tpr: 真正率
    :param title: 图表标题
    :param save_fig: 是否保存图片
    :param output_dir: 输出目录
    :return: 最大KS值
    """
    # 避免创建不必要的DataFrame，直接使用NumPy数组计算
    idx = np.arange(len(fpr)) / max(len(fpr) - 1, 1)
    ks_values = tpr - fpr
    max_ks_idx = ks_values.argmax()
    ks = round(ks_values[max_ks_idx], 4)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(idx, fpr, label='False positive rate')
    ax.plot(idx, tpr, label='True positive rate')
    ax.plot(idx, ks_values, label='KS')
    ax.axvline(idx[max_ks_idx], linestyle='--')
    ax.text(idx[max_ks_idx] + 0.03, ks - 0.1, f'KS = {ks}', color='red')
    ax.set_title(title)
    ax.legend(loc='best')
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/{title}.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    return ks


def plot_lift(y_pred, y, title, save_fig=False, output_dir="output"):
    """
    绘制Lift图（内存优化版）
    :param y_pred: 预测概率
    :param y: 真实标签
    :param title: 图表标题
    :param save_fig: 是否保存图片
    :param output_dir: 输出目录
    """
    # 使用NumPy数组优化排序和分组
    sort_indices = np.argsort(-y_pred)  # 负号用于降序
    y_sorted = y[sort_indices]
    
    n_samples = len(y_sorted)
    bin_size = n_samples // 10
    bin_indices = np.arange(0, n_samples, bin_size)
    
    bin_bad_cnt = []
    bin_obs_cnt = []
    for i in range(10):
        start = bin_indices[i]
        end = bin_indices[i + 1] if i < 9 else n_samples
        bin_data = y_sorted[start:end]
        bin_bad_cnt.append(bin_data.sum())
        bin_obs_cnt.append(len(bin_data))
    
    # 计算比率
    total_bad = sum(bin_bad_cnt)
    bad_rate = np.array(bin_bad_cnt) / total_bad
    rand_rate = np.array(bin_obs_cnt) / n_samples
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, bad_rate, width, label='Bad rate')
    ax.bar(x + width/2, rand_rate, width, label='Rand rate')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Rate')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xticks(x)
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/{title}.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)


def model_result_plot_ROC_LIFT(ds_all, tgt_col, samp_type='samp_type', 
                         save_fig=False, output_dir="output"):
    """
    模型结果汇总（含KS、AUC、PSI、ROC曲线、KS曲线、Lift图）
    :param ds_all: 完整数据集（需包含prob预测概率列）
    :param model: 模型对象
    :param tgt_col: 目标列名
    :param samp_type: 样本类型列名
    :param save_fig: 是否保存图片
    :param output_dir: 输出目录
    :param bucket_num: KS分箱数
    :return: 模型结果汇总字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动识别样本类型格式 - 支持 'train'/'01.train' 等多种格式
    unique_samp_types = ds_all[samp_type].unique()
    train_keys = [k for k in unique_samp_types if 'train' in str(k).lower()]
    test_keys = [k for k in unique_samp_types if 'test' in str(k).lower()]
    vldt_keys = [k for k in unique_samp_types if 'vldt' in str(k).lower() or 'valid' in str(k).lower() or '外推' in str(k)]
    
    # 获取各数据集数据
    train_data = ds_all[ds_all[samp_type].isin(train_keys)].reset_index(drop=True) if train_keys else pd.DataFrame()
    test_data = ds_all[ds_all[samp_type].isin(test_keys)].reset_index(drop=True) if test_keys else pd.DataFrame()
    vldt_data = ds_all[ds_all[samp_type].isin(vldt_keys)].reset_index(drop=True) if vldt_keys else pd.DataFrame()
    
    # 数据验证：确保数据不为空且标签值正确
    def validate_and_prepare(data, name):
        if len(data) == 0:
            print(f"⚠️ {name}数据为空")
            return None, None
        if 'prob' not in data.columns:
            print(f"⚠️ {name}数据缺少prob列")
            return None, None
        
        y_pred = data['prob'].values
        y = data[tgt_col].values
        
        # 确保标签值在正确范围内
        unique_vals = np.unique(y)
        if len(unique_vals) == 0:
            print(f"⚠️ {name}标签数据为空")
            return None, None
        if not set(unique_vals).issubset({0, 1, -1}):
            print(f"⚠️ {name}标签值不在{{0, 1, -1}}范围内，尝试转换")
            # 尝试转换为二分类标签
            y = (y > 0.5).astype(int)
        
        return y_pred, y
    
    # 验证和准备数据
    y_pred, y = validate_and_prepare(train_data, '训练集')
    if y_pred is None:
        raise ValueError("训练集数据无效")
    
    val_y_pred, valy = validate_and_prepare(test_data, '测试集')
    if val_y_pred is None:
        raise ValueError("测试集数据无效")
    
    off_y_pred, offy = validate_and_prepare(vldt_data, '验证集')
    if off_y_pred is None:
        raise ValueError("验证集数据无效")
    
    # 使用numpy数组进行计算
    fpr_dev, tpr_dev, _ = roc_curve(y, y_pred, drop_intermediate=True)
    fpr_val, tpr_val, _ = roc_curve(valy, val_y_pred, drop_intermediate=True)
    fpr_off, tpr_off, _ = roc_curve(offy, off_y_pred, drop_intermediate=True)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_dev, tpr_dev, label='train')
    plt.plot(fpr_val, tpr_val, label='test')
    plt.plot(fpr_off, tpr_off, label='vldt')
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

    if save_fig:
        plt.savefig(f"{output_dir}/ROC_curve.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.close()

    # 绘制KS曲线
    plot_ks(fpr_dev, tpr_dev, 'KS Curve Train', save_fig, output_dir)
    plot_ks(fpr_val, tpr_val, 'KS Curve Test', save_fig, output_dir)
    plot_ks(fpr_off, tpr_off, 'KS Curve Vldt', save_fig, output_dir)

    # 绘制Lift图
    plot_lift(y_pred, y, 'Lift Train', save_fig, output_dir)
    plot_lift(val_y_pred, valy, 'Lift Test', save_fig, output_dir)
    plot_lift(off_y_pred, offy, 'Lift Vldt', save_fig, output_dir)


######################
## 工具函数：清理临时文件
######################

def cleanup_temp_files(output_dir="output"):
    """
    清理模型报告生成的临时文件，释放内存和磁盘空间
    :param output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        return
    
    # 可以安全删除的临时文件列表
    temp_files = [
        'lr_train_ks.csv',
        'lr_test_ks.csv',
        'lr_vldt_ks.csv',
        'lr_dev_ks.csv',
        'model_ks_auc_df.csv',
        'model_psi_df.csv',
        'var_importance.csv',
        'model_var_corr.csv',
        'test_crss_prd_vld_df.csv',
        'vldt_crss_prd_vld_df.csv'
    ]
    
    deleted_count = 0
    for filename in temp_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                deleted_count += 1
            except:
                pass
    
    if deleted_count > 0:
        print(f"✅ 已清理 {deleted_count} 个临时文件，释放磁盘空间")


######################
## 报告10：跨周期稳定性（基于预测概率）（从base_code.py提取）
######################

def model_union_crss_prd_vld(ds_all, loc_var, baseline, tgt_col, output_dir, samp_type='samp_type', 
                              out_flag=False, bucket_num=10):
    """
    跨周期验证（基于预测概率）
    :param ds_all: 完整数据集（需包含prob预测概率列）
    :param loc_var: 分组变量（如年月）
    :param baseline: 基准预测值
    :param tgt_col: 目标列名
    :param samp_type: 样本类型列名
    :param out_flag: 是否输出文件
    :param bucket_num: 分箱数
    :return: 跨周期验证结果DataFrame
    """
    import warnings
    
    ym_ks = []
    ym_auc = []
    ym_psi = []
    ym_list = np.unique(ds_all[loc_var])

    for ym in ym_list:
        y_pred = ds_all[ds_all[loc_var] == ym]['prob']
        y = ds_all[ds_all[loc_var] == ym][tgt_col]
        
        # 验证数据
        if len(y_pred) == 0 or len(y) == 0:
            ym_auc.append(None)
            ym_ks.append(None)
            ym_psi.append(None)
            continue
            
        # 确保标签值正确
        y_values = y.values if hasattr(y, 'values') else np.array(y)
        unique_vals = np.unique(y_values)
        if not set(unique_vals).issubset({0, 1, -1}):
            y_values = (y_values > 0.5).astype(int)
        
        # 抑制小样本量警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            try:
                auc_val = AUC(y_pred, y_values)
                ym_auc.append(None if np.isnan(auc_val) else round(auc_val, 4))
            except:
                ym_auc.append(None)
                
            try:
                ks_val = KS(y_pred, y_values)
                ym_ks.append(None if np.isnan(ks_val) else round(ks_val, 4))
            except:
                ym_ks.append(None)
                
            try:
                psi_val = PSI(baseline, y_pred, PSI_GAP)
                ym_psi.append(None if np.isnan(psi_val) else round(psi_val, 4))
            except:
                ym_psi.append(None)
        
        if out_flag:
            try:
                toad.KS_bucket(y_pred, list(y_values), bucket=bucket_num)[[
                    'min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate',
                    'odds', 'bad_prop', 'good_prop', 'total_prop',
                    'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks'
                ]].to_csv(f"{output_dir}/{ym}_ks.csv", index=None, sep='\t')
            except:
                pass
            
    return pd.DataFrame({'ym_ks': ym_ks, 'ym_auc': ym_auc, 'ym_psi': ym_psi}, index=ym_list)


def model_union_stability(ds_all, yearmonth, tgt_col, samp_type='samp_type', output_dir="output"):
    """
    模型稳定性分析（基于预测概率）
    :param ds_all: 完整数据集（需包含prob预测概率列）
    :param yearmonth: 年月列名
    :param tgt_col: 目标列名
    :param samp_type: 样本类型列名
    :param output_dir: 输出目录
    :return: 稳定性分析结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动识别样本类型格式
    unique_samp_types = ds_all[samp_type].unique()
    train_keys = [k for k in unique_samp_types if 'train' in str(k).lower()]
    test_keys = [k for k in unique_samp_types if 'test' in str(k).lower()]
    vldt_keys = [k for k in unique_samp_types if 'vldt' in str(k).lower() or 'valid' in str(k).lower() or '外推' in str(k)]
    
    # 以训练集为基准计算跨周期PSI
    train_data = ds_all[ds_all[samp_type].isin(train_keys)].reset_index(drop=True) if train_keys else pd.DataFrame()
    cross_ym_y_pred = train_data['prob'] if len(train_data) > 0 else pd.Series(dtype=float)
    
    # print('测试集：')
    test_data = ds_all[ds_all[samp_type].isin(test_keys)].reset_index(drop=True) if test_keys else pd.DataFrame()
    test_crss_prd_vld_df = model_union_crss_prd_vld(
        test_data,
        yearmonth, cross_ym_y_pred, tgt_col, output_dir, out_flag=True, bucket_num=10
    ).reset_index()
    test_crss_prd_vld_df.to_csv(f"{output_dir}/test_crss_prd_vld_df.csv", index=False)
    
    # print('验证集：')
    vldt_data = ds_all[ds_all[samp_type].isin(vldt_keys)].reset_index(drop=True) if vldt_keys else pd.DataFrame()
    vldt_crss_prd_vld_df = model_union_crss_prd_vld(
        vldt_data,
        yearmonth, cross_ym_y_pred, tgt_col, output_dir, out_flag=True, bucket_num=10
    ).reset_index()
    vldt_crss_prd_vld_df.to_csv(f"{output_dir}/vldt_crss_prd_vld_df.csv", index=False)
    
    df1 = pd.read_csv(f"{output_dir}/test_crss_prd_vld_df.csv")
    df2 = pd.read_csv(f"{output_dir}/vldt_crss_prd_vld_df.csv")
    df1["dataset"] = "测试集"
    df2["dataset"] = "验证集"
    df = pd.concat([df1, df2])
    df = df[["index", "dataset", "ym_ks", "ym_auc", "ym_psi"]]
    df.columns = ["年月", "数据集", "KS", "AUC", "PSI"]
    
    return {
        '跨周期效果': df,
        '测试集详情': test_crss_prd_vld_df,
        '验证集详情': vldt_crss_prd_vld_df,
        '按数据集分箱：训练集': _generate_single_dataset_ks_bucket(ds_all, train_keys, samp_type, tgt_col, '训练集'),
        '按数据集分箱：测试集': _generate_single_dataset_ks_bucket(ds_all, test_keys, samp_type, tgt_col, '测试集'),
        '按数据集分箱：验证集': _generate_single_dataset_ks_bucket(ds_all, vldt_keys, samp_type, tgt_col, '验证集'),
        '年月KS分箱': _generate_yearmonth_ks_bucket(ds_all, yearmonth, tgt_col, samp_type)
    }


def _generate_single_dataset_ks_bucket(ds_all, keys, samp_type, tgt_col, dataset_name, bucket_num=10):
    """
    生成单个数据集的KS_bucket分析表
    
    参数:
        ds_all: 完整数据集
        keys: 数据集对应的样本类型键列表
        samp_type: 样本类型列名
        tgt_col: 目标列名
        dataset_name: 数据集名称
        bucket_num: 分箱数量
    
    返回:
        DataFrame: 该数据集的KS_bucket结果
    """
    if not keys:
        return pd.DataFrame()
    
    subset = ds_all[ds_all[samp_type].isin(keys)]
    if len(subset) == 0:
        return pd.DataFrame()
    
    y_pred = subset['prob']
    y = subset[tgt_col]
    
    try:
        ks_df = toad.KS_bucket(y_pred, y, bucket=bucket_num)
        # 添加分箱序号列
        ks_df.insert(0, '分箱序号', range(1, len(ks_df) + 1))
        # 数值列保留4位小数
        float_cols = ['min', 'max', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop', 'total_prop',
                      'cum_bad_rate', 'cum_bad_rate_rev', 'cum_bads_prop', 'cum_bads_prop_rev',
                      'cum_goods_prop', 'cum_goods_prop_rev', 'cum_total_prop', 'cum_total_prop_rev',
                      'ks', 'lift', 'cum_lift']
        for col in float_cols:
            if col in ks_df.columns:
                ks_df[col] = ks_df[col].round(4)
        return ks_df
    except Exception as e:
        print(f"⚠️ {dataset_name} KS_bucket计算失败: {e}")
        return pd.DataFrame()


def _generate_yearmonth_ks_bucket(ds_all, yearmonth, tgt_col, samp_type, bucket_num=10):
    """
    生成整个数据集不同年月下的KS_bucket分析表（每个年月一个DataFrame）
    
    参数:
        ds_all: 完整数据集
        yearmonth: 年月列名
        tgt_col: 目标列名
        samp_type: 样本类型列名
        bucket_num: 分箱数量
    
    返回:
        dict: {年月: DataFrame}，每个年月对应一个独立的KS_bucket结果
    """
    if yearmonth not in ds_all.columns:
        return {}
    
    ym_list = sorted(ds_all[yearmonth].dropna().unique())
    result_dict = {}
    
    for ym in ym_list:
        subset = ds_all[ds_all[yearmonth] == ym]
        if len(subset) == 0:
            continue
        
        y_pred = subset['prob']
        y = subset[tgt_col]
        
        try:
            ks_df = toad.KS_bucket(y_pred, y, bucket=bucket_num)
            # 添加分箱序号列
            ks_df.insert(0, '分箱序号', range(1, len(ks_df) + 1))
            # 数值列保留4位小数
            float_cols = ['min', 'max', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop', 'total_prop',
                          'cum_bad_rate', 'cum_bad_rate_rev', 'cum_bads_prop', 'cum_bads_prop_rev',
                          'cum_goods_prop', 'cum_goods_prop_rev', 'cum_total_prop', 'cum_total_prop_rev',
                          'ks', 'lift', 'cum_lift']
            for col in float_cols:
                if col in ks_df.columns:
                    ks_df[col] = ks_df[col].round(4)
            result_dict['按年月分箱：'+ym] = ks_df
        except Exception as e:
            print(f"⚠️ {ym} KS_bucket计算失败: {e}")
    
    return result_dict


######################
## Excel 导出（遵循excel_writer.py规范）
######################

def _write_tables_to_sheet(ws, tables, global_title=None, max_col_num=None):

    ws.sheet_view.showGridLines = False
    thin_border = Border(
        left=Side(style='thin', color='BFBFBF'), right=Side(style='thin', color='BFBFBF'),
        top=Side(style='thin', color='BFBFBF'), bottom=Side(style='thin', color='BFBFBF')
    )
    current_row = 1

    if global_title:
        # 设置行高：第一行 30
        ws.row_dimensions[current_row].height = 30
        max_col = max(len(df.columns) for _, df in tables if df is not None and len(df) > 0) if any(df is not None and len(df) > 0 for _, df in tables) else 1
        if max_col_num is not None:
            max_col = max(max_col_num, max_col)
            
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=max_col)
        cell = ws.cell(row=current_row, column=1, value=global_title)
        # 大标题 - 黄色
        cell.font = Font(name="微软雅黑", bold=True, size=12, color="FFFFFF")
        cell.fill = PatternFill(start_color="E26B0A", end_color="E26B0A", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 2

    for section_title, df in tables:
        if df is None or len(df) == 0:
            continue

        ws.row_dimensions[current_row].height = 30
        end_col = len(df.columns)
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=end_col)
        title_cell = ws.cell(row=current_row, column=1, value=section_title)
        # 小标题 - 蓝色
        title_cell.font = Font(name="微软雅黑", bold=True, size=11, color="FFFFFF")
        title_cell.fill = PatternFill(start_color="0070C0", end_color="0070C0", fill_type="solid")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1

        for col_idx, col_name in enumerate(df.columns, 1):
            cal_col_width(ws, current_row)
            ws.row_dimensions[current_row].height = 25
            cell = ws.cell(row=current_row, column=col_idx, value=col_name)
            # 列名 - 绿色
            cell.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        current_row += 1

        for _, row in df.iterrows():
            for col_idx, value in enumerate(row, 1):
                ws.row_dimensions[current_row].height = 20
                cell = ws.cell(row=current_row, column=col_idx, value=value)
                cell.font = Font(name="微软雅黑", size=10)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center", vertical="center")
            current_row += 1

        current_row += 2


def _style_single_table_sheet(ws, df, title=None, global_title=None):
    """
    对单个表格Sheet应用样式（从_write_tables_to_sheet迁移）
    :param ws: 工作表对象
    :param df: DataFrame数据
    :param title: 小标题（可选，蓝色背景白色字体）
    :param global_title: 大标题（可选，橙色背景白色字体）
    """
    # 隐藏网格线
    ws.sheet_view.showGridLines = False
    
    # 使用与 _write_tables_to_sheet 一致的边框颜色
    thin_border = Border(
        left=Side(style='thin', color='BFBFBF'),
        right=Side(style='thin', color='BFBFBF'),
        top=Side(style='thin', color='BFBFBF'),
        bottom=Side(style='thin', color='BFBFBF')
    )
    
    current_row = 1
    
    # 大标题（global_title）- 橙色背景白色字体
    if global_title:
        ws.row_dimensions[current_row].height = 30
        end_col = len(df.columns)
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=end_col)
        cell = ws.cell(row=current_row, column=1, value=global_title)
        cell.font = Font(name="微软雅黑", bold=True, size=12, color="FFFFFF")
        cell.fill = PatternFill(start_color="E26B0A", end_color="E26B0A", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 2  # 空一行
    
    # 小标题（title）- 蓝色背景白色字体
    if title:
        ws.row_dimensions[current_row].height = 30
        end_col = len(df.columns)
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=end_col)
        title_cell = ws.cell(row=current_row, column=1, value=title)
        title_cell.font = Font(name="微软雅黑", bold=True, size=11, color="FFFFFF")
        title_cell.fill = PatternFill(start_color="0070C0", end_color="0070C0", fill_type="solid")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
    
    header_row = current_row
    
    # 表头样式 - 绿色背景白色字体，10号字体
    for col_idx, col_name in enumerate(df.columns, 1):
        cal_col_width(ws, current_row)
        ws.row_dimensions[current_row].height = 25
        cell = ws.cell(row=current_row, column=col_idx, value=col_name)
        cell.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    current_row += 1
    
    # 数据行样式 - 20号行高
    for _, row in df.iterrows():
        for col_idx, value in enumerate(row, 1):
            ws.row_dimensions[current_row].height = 20
            cell = ws.cell(row=current_row, column=col_idx, value=value)
            cell.font = Font(name="微软雅黑", size=10)
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
    
    # # 冻结窗格
    # freeze_row = header_row + 1
    # ws.freeze_panes = f"A{freeze_row}"


def _add_images_to_sheet(ws, img_list, titles, start_row, start_col, output_dir="output", 
                         images_per_row=4, col_gap=1, row_gap=14):
    """
    向工作表添加图片
    :param ws: 工作表对象
    :param img_list: 图片文件名列表
    :param titles: 图片标题列表
    :param start_row: 起始行
    :param start_col: 起始列
    :param output_dir: 图片所在目录
    :param images_per_row: 每行图片数量（默认2张）
    :param col_gap: 图片标题合并单元格之间的横向间距（默认1列）
    :param row_gap: 行间距（默认14行）
    """
    std_width = 680
    std_high = 540 #480
    scale = 0.59
    col_per_image = 8 + col_gap  # 6列标题 + 间距

    for i, (img_name, title) in enumerate(zip(img_list, titles)):
        img_path = f"{output_dir}/{img_name}"
        if not os.path.exists(img_path):
            continue
        
        # 计算当前图片在行中的位置
        col_position = i % images_per_row
        row_position = i // images_per_row
        
        # 计算当前图片的起始列
        current_start_col = start_col + col_position * col_per_image
        merge_end_col = current_start_col + 7
        
        # 计算当前行
        current_row = start_row + row_position * row_gap
        
        # 合并标题单元格
        ws.merge_cells(start_row=current_row, start_column=current_start_col, 
                      end_row=current_row, end_column=merge_end_col)
        
        # 设置标题
        d_cell = ws.cell(row=current_row, column=current_start_col, value=title)
        d_cell.font = Font(name="微软雅黑", bold=True, size=11, color="FFFFFF")
        d_cell.fill = PatternFill(start_color="DC0000", end_color="DC0000", fill_type="solid")
        d_cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # 添加图片
        _add_image_to_cell(ws, img_path, std_width, std_high, scale, 
                          current_row + 1, current_start_col)

def _plot_bin_chart(var, var_data, iv, save_path):
    """
    plot_bin 风格的分箱图：
    - 左轴：堆叠柱状图（好样本/坏样本分布）
    - 右轴：bad_rate 折线
    - 显示 IV 值
    """
    # 设置中文字体，抑制字体缺失警告
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        x_list = var_data['分箱标签'].tolist()
        bad_rate = var_data['bad_rate'].values
        bin_dist = var_data['total_pct'].values
        bin_count = var_data['count'].values
        good_dist = var_data['good_pct'].values
        bad_dist = var_data['bad_pct'].values

        y_right_max = np.ceil(bad_rate.max() * 10)
        if y_right_max % 2 == 1:
            y_right_max += 1
        if y_right_max - bad_rate.max() * 10 <= 0.3:
            y_right_max += 2
        y_right_max = max(y_right_max / 10, 0.1)
        
        y_left_max = np.ceil(bin_dist.max() * 10) / 10
        y_left_max = max(y_left_max, 0.1)

        title_string = f"{var}  (iv:{round(iv, 6)})"

        ind = np.arange(len(bad_rate))
        width = 0.5

        fig, ax1 = plt.subplots(figsize=(4.5, 2.8), dpi=300)
        ax2 = ax1.twinx()

        p1 = ax1.bar(ind, good_dist, width, color=(24/255, 192/255, 196/255), label='good')
        p2 = ax1.bar(ind, bad_dist, width, bottom=good_dist, color=(246/255, 115/255, 109/255), label='bad')
        
        for i in ind:
            ax1.text(i, bin_dist[i] * 1.02, f"{round(bin_dist[i]*100,1)}%, {bin_count[i]}",
                     ha='center', fontsize=7)

        ax2.plot(ind, bad_rate, marker='o', color='blue', linewidth=1.5, markersize=5)
        for i in ind:
            ax2.text(i, bad_rate[i] * 1.02, f"{round(bad_rate[i]*100,1)}%",
                     color='blue', ha='center', fontsize=7)

        ax1.set_ylabel('Bin count distribution', fontsize=8)
        ax2.set_ylabel('Bad probability', color='blue', fontsize=8)
        ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
        ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
        ax2.tick_params(axis='y', colors='blue', labelsize=7)
        ax1.tick_params(axis='x', labelsize=7)

        plt.xticks(ind, x_list)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
        plt.title(title_string, loc='left', fontsize=10, fontweight='bold')
        ax1.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper left', fontsize=7)

        plt.tight_layout()
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)


def _add_mono_trend_charts(ws, detail_df, iv_map, output_dir):
    """
    在"单变量分箱分析" Sheet 的 G 列插入 plot_bin 风格的分箱图
    图片保存到 output_dir 目录下
    """
    if detail_df is None or len(detail_df) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    vars_list = detail_df['变量名'].unique()

    for idx, var in enumerate(vars_list):
        row_num = 3 + idx
        var_data = detail_df[detail_df['变量名'] == var].sort_values('分箱序号')
        
        if len(var_data) < 2:
            continue

        iv = iv_map.get(var, 0)
        img_path = os.path.join(output_dir, f"{var}_bin.png")
        _plot_bin_chart(var, var_data, iv, img_path)

        img = XLImage(img_path)
        img.width = 320
        img.height = 200
        ws.add_image(img, f'G{row_num}')
        ws.row_dimensions[row_num].height = 150

    ws.column_dimensions['G'].width = 45


def _write_info_sheet(ws, info_dict, title='模型基本信息'):
    """
    写入模型基本信息Sheet（采用_write_tables_to_sheet样式）
    分成两个表：基本信息表 + 模型参数表
    """
    # 隐藏网格线
    ws.sheet_view.showGridLines = False
    
    # 使用与 _write_tables_to_sheet 一致的边框颜色
    thin_border = Border(
        left=Side(style='thin', color='BFBFBF'),
        right=Side(style='thin', color='BFBFBF'),
        top=Side(style='thin', color='BFBFBF'),
        bottom=Side(style='thin', color='BFBFBF')
    )
    
    current_row = 1
    
    # 大标题 - 橙色背景白色字体
    ws.row_dimensions[current_row].height = 30
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=2)
    cell = ws.cell(row=current_row, column=1, value=title)
    cell.font = Font(name="微软雅黑", bold=True, size=12, color="FFFFFF")
    cell.fill = PatternFill(start_color="E26B0A", end_color="E26B0A", fill_type="solid")
    cell.alignment = Alignment(horizontal="center", vertical="center")
    current_row += 2  # 空一行
    
    # 分离基本信息与模型参数
    model_params_list = info_dict.get('_model_params_list', [])
    basic_info = {k: v for k, v in info_dict.items() if k != '_model_params_list'}
    
    # 表1：基本信息
    if basic_info:
        # 小标题 - 蓝色背景白色字体
        ws.row_dimensions[current_row].height = 30
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=2)
        title_cell = ws.cell(row=current_row, column=1, value='基本信息')
        title_cell.font = Font(name="微软雅黑", bold=True, size=11, color="FFFFFF")
        title_cell.fill = PatternFill(start_color="0070C0", end_color="0070C0", fill_type="solid")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
        
        # 表头
        ws.row_dimensions[current_row].height = 25
        for col_idx, col_name in enumerate(['项目', '内容'], 1):
            cell = ws.cell(row=current_row, column=col_idx, value=col_name)
            cell.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        current_row += 1
        
        # 数据行
        for key, val in basic_info.items():
            ws.row_dimensions[current_row].height = 20
            c1 = ws.cell(row=current_row, column=1, value=key)
            c2 = ws.cell(row=current_row, column=2, value=val)
            for c in [c1, c2]:
                c.font = Font(name="微软雅黑", size=10)
                c.border = thin_border
                c.alignment = Alignment(horizontal="center", vertical="center")
            current_row += 1
        
        current_row += 2  # 空二行
    
    # 表2：模型参数
    if model_params_list:
        # 小标题 - 蓝色背景白色字体
        ws.row_dimensions[current_row].height = 30
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=2)
        title_cell = ws.cell(row=current_row, column=1, value='模型参数')
        title_cell.font = Font(name="微软雅黑", bold=True, size=11, color="FFFFFF")
        title_cell.fill = PatternFill(start_color="0070C0", end_color="0070C0", fill_type="solid")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
        
        # 表头
        ws.row_dimensions[current_row].height = 25
        for col_idx, col_name in enumerate(['参数项目', '参数内容'], 1):
            cell = ws.cell(row=current_row, column=col_idx, value=col_name)
            cell.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        current_row += 1
        
        # 数据行 - 每个参数单独一行
        for param_key, param_val in model_params_list:
            ws.row_dimensions[current_row].height = 20
            c1 = ws.cell(row=current_row, column=1, value=param_key)
            c2 = ws.cell(row=current_row, column=2, value=param_val)
            for c in [c1, c2]:
                c.font = Font(name="微软雅黑", size=10)
                c.border = thin_border
                c.alignment = Alignment(horizontal="center" if c.column == 1 else "center", vertical="center")
            current_row += 1
    
    # 设置列宽
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 50


def _add_image_to_cell(ws, img_path, std_width, std_high, scale, row_num, col_num):
    """
    向单元格添加图片
    :param ws: 工作表对象
    :param img_path: 图片路径
    :param std_width: 标准宽度
    :param std_high: 标准高度
    :param scale: 缩放比例
    :param row_num: 行号
    :param col_num: 列号（从1开始）
    """
    from PIL import Image as PILImage
    from openpyxl.utils import get_column_letter
    image = PILImage.open(img_path)
    x_scale = std_width * scale / image.size[0]
    y_scale = std_high * scale / image.size[1]
    img = XLImage(img_path)
    img.width = image.size[0] * x_scale
    img.height = image.size[1] * y_scale
    col_letter = get_column_letter(col_num)
    ws.add_image(img, f'{col_letter}{row_num}')

def export_report_to_excel(report_dict, output_path, output_dir="output"):
    """
    导出完整报告到Excel文件（遵循excel_writer.py规范）
    :param report_dict: 报告数据字典
    :param output_path: 输出文件路径
    :param output_dir: 图片目录
    :param output_dir: 输出目录
    """
    
    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        workbook = writer.book

        # Sheet1: 模型基本信息
        ws_info = workbook.create_sheet(title='模型基本信息')
        _write_info_sheet(ws_info, report_dict['模型基本信息'], title='模型基本信息')
        # _apply_global_style(ws_info)

        # Sheet2: 模型综合效果
        r01 = report_dict['模型综合效果']
        tables = [
            ('样本效果（训练/测试/验证）', r01['模型效果汇总']),
            ('模型效果（PSI）', r01['模型效果_PSI']),
        ]
        if r01['各月分布'] is not None:
            tables.insert(3, ('各月样本分布', r01['各月分布']))
            tables.insert(4, ('各月样本分布汇总', r01['各月分布_汇总']))
        tables.append(('LIFT 分箱明细', r01['LIFT分箱']))

        ws_model = workbook.create_sheet(title='模型综合效果')
        _write_tables_to_sheet(ws=ws_model, tables=tables, global_title='模型综合效果报告', max_col_num=44)
        # 添加图片
        img_list = [
            'ROC_curve.png',
            'KS Curve Train.png',
            'KS Curve Test.png',
            'KS Curve Vldt.png',
            'Lift Train.png',
            'Lift Test.png',
            'Lift Vldt.png'
        ]
        titles = [
            '模型 ROC 曲线',
            '训练集 KS 曲线',
            '测试集 KS 曲线',
            '验证集 KS 曲线',
            '训练集 Lift',
            '测试集 Lift',
            '验证集 Lift'
        ]
        _add_images_to_sheet(ws_model, img_list, titles, 3, 10, output_dir)
        for col_num in range(10, 44 + 1):
            col_letter = get_column_letter(col_num)
            ws_model.column_dimensions[col_letter].width = 7


        # Sheet3: 入模变量信息
        report_dict['入模变量信息'].to_excel(writer, sheet_name='入模变量信息', index=False)
        _style_single_table_sheet(workbook['入模变量信息'], report_dict['入模变量信息'], title='入模变量基本信息（IV / 重要性 / PSI）')
        # _apply_global_style(workbook['入模变量信息'])

        # Sheet4: 单变量分箱分析（含 plot_bin 风格图）
        summary_df = report_dict['变量分箱_是否单调']['汇总'].copy()
        
        # 单独提取 IV 映射，不写入 Excel
        iv_map = {}
        if 'IV' in report_dict['入模变量信息'].columns:
            iv_df = report_dict['入模变量信息'][['变量名', 'IV']].copy()
            iv_map = dict(zip(iv_df['变量名'], iv_df['IV']))
        
        # 确保 summary_df 只有6列，不要有多余列占用 G
        keep_cols = ['变量名', '是否单调', '单调趋势', '分箱数', '最小bad_rate', '最大bad_rate']
        summary_df = summary_df[[c for c in keep_cols if c in summary_df.columns]]
        summary_df['分箱分布图'] = ''  # 第一个空列
        summary_df['分组违约率趋势图'] = ''  # 第二个空列
        summary_df.to_excel(writer, sheet_name='单变量分箱分析', index=False)
        _style_single_table_sheet(workbook['单变量分箱分析'], summary_df, title='单变量分箱分析')
        _add_mono_trend_charts(
            workbook['单变量分箱分析'],
            report_dict['变量分箱_是否单调']['分箱明细'],
            iv_map,
            output_dir=output_dir
        )
        ws_mono = workbook['单变量分箱分析']
        vars_list = summary_df['变量名'].unique() if '变量名' in summary_df.columns else []
        if '变量分箱图表' in report_dict and 'plot_badrate_data' in report_dict['变量分箱图表']:
            plot_badrate_data = report_dict['变量分箱图表']['plot_badrate_data']
            badrate_map = {item['col']: item for item in plot_badrate_data}
            for idx, var_name in enumerate(vars_list):
                row_num = 3 + idx
                if var_name in badrate_map:
                    item = badrate_map[var_name]
                    img_path = os.path.join(output_dir, f"{var_name}_plot_badrate.png")
                    try:
                        plot_badrate(item['df_bin'], item['by_col'], var_name, item['tgt_col'], save_path=img_path)
                        
                        if os.path.exists(img_path):
                            img = XLImage(img_path)
                            img.width = 320
                            img.height = 200
                            ws_mono.add_image(img, f'H{row_num}')
                    except Exception as e:
                        print(f"⚠️ 绘制 {var_name} 的 plot_badrate 图表失败: {e}")
        
        # 设置H列宽
        ws_mono.column_dimensions['H'].width = 45

        # Sheet6: 入模变量概况（新增）
        if '入模变量概况' in report_dict:
            report_dict['入模变量概况'].to_excel(writer, sheet_name='入模变量概况', index=False)
            _style_single_table_sheet(workbook['入模变量概况'], report_dict['入模变量概况'], title='入模变量概况（IV / Gini / Entropy / PSI）')
            # _apply_global_style(workbook['入模变量概况'])

        # Sheet7: 模型稳定性（新增）
        if '模型稳定性' in report_dict:
            ws_stability = workbook.create_sheet(title='模型稳定性')
            stability_data = [
                ('模型跨周期效果（PSI基准为全部训练集）', report_dict['模型稳定性']['跨周期效果'])
            ]
            # 添加数据集KS分箱表（每个数据集一个表）
            for ds_key in ['按数据集分箱：训练集', '按数据集分箱：测试集', '按数据集分箱：验证集']:
                if ds_key in report_dict['模型稳定性'] and len(report_dict['模型稳定性'][ds_key]) > 0:
                    stability_data.append((ds_key, report_dict['模型稳定性'][ds_key]))
            # 添加年月KS分箱表（每个年月一个表）
            if '年月KS分箱' in report_dict['模型稳定性'] and isinstance(report_dict['模型稳定性']['年月KS分箱'], dict):
                for ym, ym_df in report_dict['模型稳定性']['年月KS分箱'].items():
                    if len(ym_df) > 0:
                        stability_data.append((f'{ym}', ym_df))
            _write_tables_to_sheet(ws_stability, stability_data, global_title='模型稳定性分析')

        # Sheet9: 变量相关性分析（新增）
        if '变量相关性分析' in report_dict:
            ws_corr = workbook.create_sheet(title='变量相关性分析')
            tables = []
            # 添加相关性矩阵表
            if '相关性矩阵' in report_dict['变量相关性分析']:
                tables.append(('相关性矩阵', report_dict['变量相关性分析']['相关性矩阵']))
            # 添加相关性Top10表
            if '相关性Top10' in report_dict['变量相关性分析']:
                tables.append(('相关性Top10', report_dict['变量相关性分析']['相关性Top10']))
            _write_tables_to_sheet(ws_corr, tables, global_title='变量相关性分析')

    print(f"✅ 模型报告已导出至: {output_path}")
    print(f"✅ 临时文件已保存至: {output_dir}")


######################
## 主函数
######################

def model_report_main(model, model_info, model_data, y_flag, ym_flag=None, data_flag="samp_type", 
                      output_excel=None, output_dir="output", 
                      cleanup_temp=True, pkl_path=None):
    """
    模型报告主函数 - 生成base_code.py中的所有报告
    :param model: 训练好的模型对象
    :param model_info: 模型基本信息字典
    :param model_data: 包含特征和标签的DataFrame
    :param y_flag: 标签列名
    :param ym_flag: 年月列名（可选）
    :param data_flag: 数据类型列名（默认"samp_type"）
    :param output_excel: 输出Excel路径（可选）
    :param output_dir: 输出目录（可选）
    :param cleanup_temp: 是否清理临时文件（默认True）
    :return: 完整报告字典
    """
    work_data = model_data.copy()
    output_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(output_dir,exist_ok=True)
    model_vars = model.booster_.feature_name()
    model_pred = model.predict_proba(work_data[model_vars])[:, 1]
    work_data['model_pred'] = model_pred
    work_data['prob'] = model_pred  # 用于兼容性
    pred_flag = 'model_pred'

    full_report = { '模型基本信息': model_info }
    try:
        # 获取模型参数 - 存储为列表，每个参数是独立的字典项
        model_params = model.get_params()
        model_info['_model_params_list'] = [
            (k, v) for k, v in model_params.items()
            if v is not None and not isinstance(v, (dict, list))
        ]
        # 添加入模变量数
        target_key = 'vldt 年月集'
        new_dict = {}
        for k, v in model_info.items():
            if k == target_key:
                new_dict['入模变量数'] = len(model_vars)
            new_dict[k] = v
        model_info = new_dict
        # 添加样本数量信息
        if data_flag in work_data.columns:
            for samp in work_data[data_flag].unique():
                cnt = len(work_data[work_data[data_flag] == samp])
                model_info[f'{samp} 样本数'] = cnt
        model_info['模型文件'] = ''
        full_report['模型基本信息'] = model_info
    except Exception as e:
        print(f"⚠️ 增强模型基本信息失败: {e}")

    # 基础报告 1
    try:
        report01 = model_report_01(work_data, pred_flag, y_flag,data_flag=data_flag, ym_flag=ym_flag,n_bins=10, ascending=False)
        full_report['模型综合效果'] = report01
        print("✅ 样本模型综合效果报告已生成")
    except Exception as e:
        print(f"⚠️ 样本模型综合效果报告生成失败: {e}")
    
    # 基础报告 2
    try:
        report03 = model_report_03(work_data, model, y_flag, ym_flag, samp_type=data_flag)
        model_result_plot_ROC_LIFT(
            work_data, y_flag, samp_type=data_flag,
            save_fig=True, output_dir=output_dir
        )
        full_report['入模变量信息'] = report03
        print("✅ 样本入模变量信息报告已生成")
    except Exception as e:
        print(f"⚠️ 样本入模变量信息报告生成失败: {e}")
    
     # 基础报告 3
    try:
        report04 = model_report_04(work_data, model, y_flag, n_bins=10)
        full_report['变量分箱_是否单调'] = report04
        print("✅ 样本变量分箱_是否单调报告已生成")
    except Exception as e:
        print(f"⚠️ 样本变量分箱_是否单调报告生成失败: {e}")

    # 报告10: 模型稳定性（基于预测概率）
    try:
        stability = model_union_stability(work_data, ym_flag or 'yearmonth', y_flag, samp_type=data_flag, output_dir=output_dir)
        full_report['模型稳定性'] = stability
        print("✅ 模型稳定性报告已生成")
    except Exception as e:
        print(f"⚠️ 模型稳定性报告生成失败: {e}")

    # 报告11: 变量相关性分析（新增）
    try:
        train_data = work_data[work_data[data_flag] == 'train'] if 'train' in work_data[data_flag].unique() else work_data
        corr_result = model_var_corr(train_data[model_vars], output_dir=output_dir)
        full_report['变量相关性分析'] = corr_result
        print("✅ 变量相关性分析报告已生成")
    except Exception as e:
        print(f"⚠️ 变量相关性分析报告生成失败: {e}")

    # 报告12: 变量分箱图表（plot_bin 和 plot_badrate）
    try:
        chart_data = {'plot_bin_data': {}, 'plot_badrate_data': []}
        
        # 获取分箱明细数据用于 plot_bin
        if '变量分箱_是否单调' in full_report and '分箱明细' in full_report['变量分箱_是否单调']:
            bin_detail = full_report['变量分箱_是否单调']['分箱明细']
            
            # 为每个变量准备分箱数据
            for var_name in model_vars:
                if var_name not in work_data.columns:
                    continue
                    
                # 获取该变量的原始数据和标签
                df_var = work_data[[var_name, y_flag]].dropna()
                if len(df_var) == 0:
                    continue
                
                # 使用 toad 进行分箱
                try:
                    c = toad.transform.Combiner()
                    c.fit(df_var[[var_name, y_flag]], y=y_flag, method='chi', n_bins=min(10, len(df_var)//2))
                    bins = c.export()[var_name]
                    df_binned = pd.DataFrame({var_name: pd.cut(df_var[var_name], bins=bins, include_lowest=True)})
                    df_y_aligned = df_var[y_flag].reset_index(drop=True)
                    
                    chart_data['plot_bin_data'][var_name] = {
                        'df_bin': df_binned,
                        'df_y': df_y_aligned
                    }
                except Exception as bin_err:
                    # 如果 chi 分箱失败，使用 qcut
                    try:
                        df_var_reset = df_var.reset_index(drop=True)
                        df_binned = pd.DataFrame({var_name: pd.qcut(df_var_reset[var_name], q=10, labels=False, duplicates='drop')})
                        df_y_aligned = df_var_reset[y_flag]
                        chart_data['plot_bin_data'][var_name] = {
                            'df_bin': df_binned,
                            'df_y': df_y_aligned
                        }
                    except:
                        pass
        
        # 如果有年月字段，生成 plot_badrate 数据（需要先对变量进行分箱）
        if ym_flag and ym_flag in work_data.columns:
            for var_name in model_vars:
                if var_name not in work_data.columns:
                    continue
                
                # 获取该变量的原始数据
                df_var = work_data[[ym_flag, var_name, y_flag]].dropna()
                if len(df_var) == 0:
                    continue
                
                # 使用 toad 进行分箱
                try:
                    c = toad.transform.Combiner()
                    c.fit(df_var[[var_name, y_flag]], y=y_flag, method='chi', n_bins=min(10, len(df_var)//2))
                    bins = c.export()[var_name]
                    df_binned = df_var.copy()
                    df_binned[var_name] = pd.cut(df_binned[var_name], bins=bins, include_lowest=True)
                    
                    chart_data['plot_badrate_data'].append({
                        'df_bin': df_binned,
                        'by_col': ym_flag,
                        'col': var_name,
                        'tgt_col': y_flag
                    })
                except Exception as bin_err:
                    # 如果 chi 分箱失败，使用 qcut
                    try:
                        df_var_reset = df_var.reset_index(drop=True)
                        df_binned = df_var_reset.copy()
                        df_binned[var_name] = pd.qcut(df_binned[var_name], q=10, labels=False, duplicates='drop')
                        chart_data['plot_badrate_data'].append({
                            'df_bin': df_binned,
                            'by_col': ym_flag,
                            'col': var_name,
                            'tgt_col': y_flag
                        })
                    except:
                        pass
        
        full_report['变量分箱图表'] = chart_data
        print("✅ 变量分箱图表数据已生成")
    except Exception as e:
        print(f"⚠️ 变量分箱图表数据生成失败: {e}")

    if output_excel:
        export_report_to_excel(full_report, output_excel, output_dir=output_dir)
    
    export_pmml_and_insert_to_excel(
        model=model,
        pkl_path=pkl_path,
        excel_path=output_excel,
        sheet_name='模型基本信息'
    )
    
    # 清理临时文件
    # if cleanup_temp:
    #     cleanup_temp_files(output_dir=output_dir)

    return full_report


# ==================== PMML导出与OLE对象插入功能 ====================

def export_model_to_pmml(model, pkl_path, pmml_path=None, with_repr=True):
    if not HAS_SKLEARN2PMML:
        print("⚠️ sklearn2pmml未安装，无法导出PMML")
        return None
    
    if pmml_path is None:
        pmml_path = pkl_path.replace('.pkl', '.pmml') if pkl_path else 'model.pmml'
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(pmml_path) if os.path.dirname(pmml_path) else '.', exist_ok=True)
        
        # 导出PMML
        sklearn2pmml(estimator=model, pmml_path=pmml_path, with_repr=with_repr)
        print(f"✅ PMML文件已导出: {pmml_path}")
        return pmml_path
    except Exception as e:
        print(f"⚠️ PMML导出失败: {e}")
        return None


def insert_ole_object(excel_path, sheet_name, cell, file_path, display_as_icon=True, label=None):
    if not HAS_WIN32:
        print("⚠️ pywin32未安装，无法插入OLE对象")
        return False
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return False
    tmp_dir = tempfile.gettempdir()
    _, ext = os.path.splitext(file_path)
    if label:
        tmp_file_path = os.path.join(tmp_dir, f"{label}{ext}")
    else:
        tmp_file_path = os.path.join(tmp_dir, os.path.basename(file_path))
    try:
        shutil.copy2(file_path, tmp_file_path)
    except Exception as e:
        print(f"⚠️ 文件复制失败: {e}")
        return False
    
    try:
        excel = win32.DispatchEx("Excel.Application")
        excel = win32.gencache.EnsureDispatch(excel)
        excel.Visible = False
        
        wb = excel.Workbooks.Open(os.path.abspath(excel_path))
        ws = wb.Sheets(sheet_name)
        
        ole_obj = ws.OLEObjects().Add(
            Filename=os.path.abspath(tmp_file_path),
            Link=False,
            DisplayAsIcon=display_as_icon,
            Left=40 + ws.Range('B13').Left + cell,
            Top=10 + ws.Range('B13').Top
        )
        ole_obj.Border.LineStyle = 0
            
        wb.Save()
        wb.Close()
        excel.Quit()
        
        print(f"✅ OLE对象已插入: {file_path} -> B13")
        return True
    except Exception as e:
        print(f"⚠️ OLE对象插入失败: {e}")
        try:
            excel.Quit()
        except:
            pass
        return False


def _is_windows_with_excel():
    """
    检查是否是Windows系统且安装了Excel
    
    返回:
        bool: True表示是Windows且有Excel，False表示不是
    """
    import platform
    if platform.system() != 'Windows':
        return False
    
    if not HAS_WIN32:
        return False
    
    try:
        excel = win32.DispatchEx("Excel.Application")
        excel.Quit()
        return True
    except:
        return False


def _adjust_sheet_row_for_ole(excel_path, sheet_name, row_num=13, row_height=60):
    """
    调整Excel指定行的行高
    
    参数:
        excel_path: Excel文件路径
        sheet_name: 工作表名称
        row_num: 行号（默认13）
        row_height: 行高（默认60）
    
    返回:
        bool: 成功返回True，失败返回False
    """
    try:
        from openpyxl import load_workbook
        wb = load_workbook(excel_path)
        if sheet_name not in wb.sheetnames:
            wb.close()
            return False
        ws = wb[sheet_name]
        ws.row_dimensions[row_num].height = row_height
        wb.save(excel_path)
        wb.close()
        return True
    except Exception as e:
        print(f"⚠️ 调整行高失败: {e}")
        return False


def _delete_sheet_row(excel_path, sheet_name, row_num=13):
    """
    删除Excel指定行
    
    参数:
        excel_path: Excel文件路径
        sheet_name: 工作表名称
        row_num: 行号（默认13）
    
    返回:
        bool: 成功返回True，失败返回False
    """
    try:
        from openpyxl import load_workbook
        wb = load_workbook(excel_path)
        if sheet_name not in wb.sheetnames:
            wb.close()
            return False
        ws = wb[sheet_name]
        ws.delete_rows(row_num)
        wb.save(excel_path)
        wb.close()
        return True
    except Exception as e:
        print(f"⚠️ 删除行失败: {e}")
        return False


def export_pmml_and_insert_to_excel(model, pkl_path, excel_path, sheet_name='模型基本信息',
                                     cell1='D5', cell2='D10', pmml_path=None, with_repr=True,
                                     display_as_icon=True):

    result = {'pmml_path': None, 'success': False}
    
    # 1. 导出PMML
    pmml_path = export_model_to_pmml(model, pkl_path, pmml_path, with_repr)
    if pmml_path is None:
        return result
    
    result['pmml_path'] = pmml_path
    
    # 2. 检查Windows和Excel环境
    if not _is_windows_with_excel():
        print("⚠️ 非Windows系统或未安装Excel，跳过OLE对象插入")
        # 删除第13行
        if os.path.exists(excel_path):
            _delete_sheet_row(excel_path, sheet_name, row_num=13)
        return result
    
    # 3. 插入OLE对象到Excel
    if os.path.exists(excel_path):
        # 先调整第13行行宽
        _adjust_sheet_row_for_ole(excel_path, sheet_name, row_num=13, row_height=60)
        
        success1 = insert_ole_object(
            excel_path, sheet_name, 0, pkl_path,
            display_as_icon=display_as_icon,
            label='pkl模型文件'
        )
        success2 = insert_ole_object(
            excel_path, sheet_name, 80, pmml_path,
            display_as_icon=display_as_icon,
            label='pmml模型文件'
        )
        result['success'] = success1 and success2
    else:
        print(f"⚠️ Excel文件不存在: {excel_path}")
    
    return result
