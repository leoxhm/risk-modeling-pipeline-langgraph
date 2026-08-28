import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#########################
# step01-样本分布模块
#########################
def model_data_total_info(model_data, y_flag, 
                          month_flag=None, 
                          group_flag=None, 
                          channel_flag=None):
    """
    查看样本整体分布：好坏占比、总量、各客群/各月份/各渠道占比
    支持两两交叉：客群-月份、客群-渠道、渠道-月份
    返回字典，不打印
    """
    results = {}
    
    # 1. 整体样本情况
    total_cnt = len(model_data)
    good_cnt = (model_data[y_flag] == 0).sum()
    bad_cnt = (model_data[y_flag] == 1).sum()
    bad_rate = bad_cnt / total_cnt
    
    results['总体'] = pd.DataFrame([{
        '样本总量': total_cnt,
        '好样本数': good_cnt,
        '坏样本数': bad_cnt,
        '好样本占比': f"{good_cnt/total_cnt:.2%}",
        '坏样本占比': f"{bad_cnt/total_cnt:.2%}",
        '坏样本率': f"{bad_rate:.2%}"
    }])
    
    # 2. 按月份分布
    if month_flag and month_flag in model_data.columns:
        month_stats = []
        for month in sorted(model_data[month_flag].unique()):
            month_data = model_data[model_data[month_flag] == month]
            month_total_cnt = len(month_data)
            month_good_cnt = (month_data[y_flag] == 0).sum()
            month_bad_cnt = (month_data[y_flag] == 1).sum()
            month_bad_rate = month_bad_cnt / month_total_cnt
            
            month_stats.append({
                '月份': month,
                '样本总量': month_total_cnt,
                '好样本数': month_good_cnt,
                '坏样本数': month_bad_cnt,
                '好样本占比': f"{month_good_cnt/total_cnt:.2%}",
                '坏样本占比': f"{month_bad_cnt/total_cnt:.2%}",
                '样本占比': f"{month_total_cnt/total_cnt:.2%}",
                '坏样本率': f"{month_bad_rate:.2%}"
            })
        
        results['月份分布'] = pd.DataFrame(month_stats)
    
    # 3. 按客群分布
    if group_flag and group_flag in model_data.columns:
        group_stats = []
        for group in sorted(model_data[group_flag].unique()):
            group_data = model_data[model_data[group_flag] == group]
            group_total_cnt = len(group_data)
            group_good_cnt = (group_data[y_flag] == 0).sum()
            group_bad_cnt = (group_data[y_flag] == 1).sum()
            group_bad_rate = group_bad_cnt / group_total_cnt
            
            group_stats.append({
                '客群': group,
                '样本总量': group_total_cnt,
                '好样本数': group_good_cnt,
                '坏样本数': group_bad_cnt,
                '好样本占比': f"{group_good_cnt/total_cnt:.2%}",
                '坏样本占比': f"{group_bad_cnt/total_cnt:.2%}",
                '样本占比': f"{group_total_cnt/total_cnt:.2%}",
                '坏样本率': f"{group_bad_rate:.2%}"
            })
        
        results['客群分布'] = pd.DataFrame(group_stats)
    
    # 4. 按渠道分布
    if channel_flag and channel_flag in model_data.columns:
        channel_stats = []
        for channel in sorted(model_data[channel_flag].unique()):
            channel_data = model_data[model_data[channel_flag] == channel]
            channel_total_cnt = len(channel_data)
            channel_good_cnt = (channel_data[y_flag] == 0).sum()
            channel_bad_cnt = (channel_data[y_flag] == 1).sum()
            channel_bad_rate = channel_bad_cnt / channel_total_cnt
            
            channel_stats.append({
                '渠道': channel,
                '样本总量': channel_total_cnt,
                '好样本数': channel_good_cnt,
                '坏样本数': channel_bad_cnt,
                '好样本占比': f"{channel_good_cnt/total_cnt:.2%}",
                '坏样本占比': f"{channel_bad_cnt/total_cnt:.2%}",
                '样本占比': f"{channel_total_cnt/total_cnt:.2%}",
                '坏样本率': f"{channel_bad_rate:.2%}"
            })
        
        results['渠道分布'] = pd.DataFrame(channel_stats)
    
    # 5. 两两交叉分布（客群在前，月份/渠道在后）
    
    # 客群-月份组合（按客群分组，月份在后）
    if group_flag and month_flag and group_flag in model_data.columns and month_flag in model_data.columns:
        cross_stats = []
        for group in sorted(model_data[group_flag].unique()):
            for month in sorted(model_data[month_flag].unique()):
                cross_data = model_data[(model_data[group_flag] == group) & (model_data[month_flag] == month)]
                if len(cross_data) == 0:
                    continue
                    
                cross_total_cnt = len(cross_data)
                cross_good_cnt = (cross_data[y_flag] == 0).sum()
                cross_bad_cnt = (cross_data[y_flag] == 1).sum()
                cross_bad_rate = cross_bad_cnt / cross_total_cnt
                
                cross_stats.append({
                    '客群': group,
                    '月份': month,
                    '样本总量': cross_total_cnt,
                    '好样本数': cross_good_cnt,
                    '坏样本数': cross_bad_cnt,
                    '好样本占比': f"{cross_good_cnt/total_cnt:.2%}",
                    '坏样本占比': f"{cross_bad_cnt/total_cnt:.2%}",
                    '样本占比': f"{cross_total_cnt/total_cnt:.2%}",
                    '坏样本率': f"{cross_bad_rate:.2%}"
                })
        
        results['客群_月份'] = pd.DataFrame(cross_stats)
    
    # 客群-渠道组合（按客群分组，渠道在后）
    if group_flag and channel_flag and group_flag in model_data.columns and channel_flag in model_data.columns:
        cross_stats = []
        for group in sorted(model_data[group_flag].unique()):
            for channel in sorted(model_data[channel_flag].unique()):
                cross_data = model_data[(model_data[group_flag] == group) & (model_data[channel_flag] == channel)]
                if len(cross_data) == 0:
                    continue
                    
                cross_total_cnt = len(cross_data)
                cross_good_cnt = (cross_data[y_flag] == 0).sum()
                cross_bad_cnt = (cross_data[y_flag] == 1).sum()
                cross_bad_rate = cross_bad_cnt / cross_total_cnt
                
                cross_stats.append({
                    '客群': group,
                    '渠道': channel,
                    '样本总量': cross_total_cnt,
                    '好样本数': cross_good_cnt,
                    '坏样本数': cross_bad_cnt,
                    '好样本占比': f"{cross_good_cnt/total_cnt:.2%}",
                    '坏样本占比': f"{cross_bad_cnt/total_cnt:.2%}",
                    '样本占比': f"{cross_total_cnt/total_cnt:.2%}",
                    '坏样本率': f"{cross_bad_rate:.2%}"
                })
        
        results['客群_渠道'] = pd.DataFrame(cross_stats)
    
    # 渠道-月份组合（按渠道分组，月份在后）
    if channel_flag and month_flag and channel_flag in model_data.columns and month_flag in model_data.columns:
        cross_stats = []
        for channel in sorted(model_data[channel_flag].unique()):
            for month in sorted(model_data[month_flag].unique()):
                cross_data = model_data[(model_data[channel_flag] == channel) & (model_data[month_flag] == month)]
                if len(cross_data) == 0:
                    continue
                    
                cross_total_cnt = len(cross_data)
                cross_good_cnt = (cross_data[y_flag] == 0).sum()
                cross_bad_cnt = (cross_data[y_flag] == 1).sum()
                cross_bad_rate = cross_bad_cnt / cross_total_cnt
                
                cross_stats.append({
                    '渠道': channel,
                    '月份': month,
                    '样本总量': cross_total_cnt,
                    '好样本数': cross_good_cnt,
                    '坏样本数': cross_bad_cnt,
                    '好样本占比': f"{cross_good_cnt/total_cnt:.2%}",
                    '坏样本占比': f"{cross_bad_cnt/total_cnt:.2%}",
                    '样本占比': f"{cross_total_cnt/total_cnt:.2%}",
                    '坏样本率': f"{cross_bad_rate:.2%}"
                })
        
        results['渠道_月份'] = pd.DataFrame(cross_stats)
    
    return results


def print_df_table(df, title=None, group_cols=None):
    """
    通用DataFrame打印函数，自动对齐，支持分组空行
    
    Parameters:
    -----------
    df : DataFrame
        要打印的数据
    title : str, optional
        表格标题
    group_cols : list, optional
        需要分组空行的列名列表（按这些列变化时插入空行）
    """
    if df is None or len(df) == 0:
        return
    
    # 打印标题
    if title:
        width = 80
        print(f"\n{'='*width}")
        print(f"{title:^{width}}")
        print(f"{'='*width}")
    
    # 获取列名
    cols = df.columns.tolist()
    
    # 计算每列宽度（中文字符算2个宽度）
    def str_width(s):
        return sum(2 if ord(c) > 127 else 1 for c in str(s))
    
    col_widths = {}
    for col in cols:
        header_w = str_width(col)
        max_data_w = df[col].astype(str).apply(str_width).max()
        col_widths[col] = max(header_w, max_data_w) + 2  # 加2空格缓冲
    
    # 构建分隔线
    sep_line = "+".join(["-" * (col_widths[col] + 2) for col in cols])
    sep_line = "+" + sep_line + "+"
    
    # 打印表头
    print(sep_line)
    header_cells = [f" {str(col):^{col_widths[col]}} " for col in cols]
    print("|" + "|".join(header_cells) + "|")
    print(sep_line)
    
    # 打印数据
    prev_group_vals = None if group_cols is None else [None] * len(group_cols)
    
    for idx, row in df.iterrows():
        # 检查是否需要插入空行（分组）
        if group_cols and prev_group_vals[0] is not None:
            current_vals = [row[col] for col in group_cols]
            if current_vals[0] != prev_group_vals[0]:
                print(sep_line)
        
        # 打印数据行
        row_cells = []
        for col in cols:
            val = str(row[col])
            pad = col_widths[col] - str_width(val)
            left_pad = pad // 2
            right_pad = pad - left_pad
            cell = " " * left_pad + val + " " * right_pad
            row_cells.append(f" {cell} ")
        
        print("|" + "|".join(row_cells) + "|")
        
        # 更新分组值
        if group_cols:
            prev_group_vals = [row[col] for col in group_cols]
    
    print(sep_line)





def create_sample_data(n_customers=5000, start_date='2023-01-01', months=12, random_state=42):
    """
    创建模拟数据用于LGB模型训练
    
    Parameters:
    -----------
    n_customers: 客户数量
    start_date: 开始日期
    months: 月份数
    random_state: 随机种子
    
    Returns:
    --------
    DataFrame: 包含主键、特征、月份和Y标签的数据
    """
    np.random.seed(random_state)
    
    # 生成客户ID
    customer_ids = [f'CUST_{i+1:05d}' for i in range(n_customers)]
    
    # 生成日期序列
    start_date = pd.to_datetime(start_date)
    dates = [start_date + timedelta(days=30*i) for i in range(months)]
    
    # 创建数据列表
    data_list = []
    
    # 为每个客户生成基础特征
    for customer_id in customer_ids:
        # 客户基础特征（固定）
        base_features = {
            'feature_1': np.random.normal(50, 15),
            'feature_2': np.random.choice([0, 1], p=[0.3, 0.7]),
            'feature_3': np.random.exponential(1000),
            'feature_4': np.random.randint(300, 850),
            'feature_5': np.random.choice(['A', 'B', 'C', 'D'], p=[0.4, 0.3, 0.2, 0.1]),
            'feature_6': np.random.poisson(5),
            'feature_7': np.random.uniform(0, 1),
            'feature_8': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),
            'feature_9': np.random.gamma(2, 2),
            'feature_10': np.random.randint(0, 100)
        }
        
        # 为每个月份生成数据
        for month_idx, month_date in enumerate(dates):
            month = month_date.month
            quarter = (month - 1) // 3 + 1
            time_trend = month_idx / months
            seasonal = np.sin(2 * np.pi * month / 12)
            
            row = {
                'primary_key': f"{customer_id}_{month_date.strftime('%Y%m')}",
                'customer_id': customer_id,
                'year_month': month_date.strftime('%Y-%m'),
                'month': month,
                'quarter': quarter,
                'year': month_date.year,
                
                # 基础特征
                'feature_1': base_features['feature_1'] + np.random.normal(0, 2),
                'feature_2': base_features['feature_2'],
                'feature_3': base_features['feature_3'] * (1 + time_trend * 0.1) + np.random.normal(0, 50),
                'feature_4': base_features['feature_4'] + np.random.randint(-20, 20),
                'feature_5': base_features['feature_5'],
                'feature_6': max(0, base_features['feature_6'] + np.random.poisson(2) + int(seasonal * 3)),
                'feature_7': min(1, max(0, base_features['feature_7'] + np.random.normal(0, 0.05))),
                'feature_8': base_features['feature_8'],
                'feature_9': max(0, base_features['feature_9'] * (1 + seasonal * 0.2) + np.random.normal(0, 1)),
                'feature_10': min(100, max(0, base_features['feature_10'] + np.random.randint(-10, 10))),
            }
            
            # 生成目标变量Y
            logit = (
                -3.5
                + 0.02 * row['feature_1']
                - 0.5 * row['feature_2']
                - 0.001 * row['feature_3']
                - 0.005 * row['feature_4']
                + (1 if row['feature_5'] == 'D' else 0) * 1.2
                + 0.05 * row['feature_6']
                + 2 * row['feature_7']
                + (1 if row['feature_8'] == 2 else 0) * 0.8
                + 0.01 * row['feature_9']
                - 0.01 * row['feature_10']
                + time_trend * 0.5
                + seasonal * 0.3
            )
            
            prob = 1 / (1 + np.exp(-logit))
            prob = np.clip(prob + np.random.normal(0, 0.05), 0, 1)
            row['Y'] = 1 if np.random.random() < prob else 0
            
            data_list.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 添加缺失值
    for col in ['feature_1', 'feature_3', 'feature_6', 'feature_9']:
        missing_idx = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

