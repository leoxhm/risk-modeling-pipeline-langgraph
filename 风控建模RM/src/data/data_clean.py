# !/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import pandas as pd
import toad
import numpy as np
from src.utils.logger import setup_logging,log_progress
logger = setup_logging('data_clean')



"""
    时间清洗模块
"""
def data_time_change(data_time):
    """
    :param data_time: 日期
    :return:处理后日期
    """
    if "/" in data_time:
        year = data_time[:4]
        if len(data_time[data_time.rfind('/') + 1:]) == 1:
            day = '0' + data_time[data_time.rfind('/') + 1]
        else:
            day = data_time[data_time.rfind('/') + 1:]
        if len(data_time[data_time.find('/') + 1: data_time.rfind('/')]) == 1:
            month = '0' + data_time[data_time.find('/') + 1:data_time.rfind('/')]
        else:
            month = data_time[data_time.find('/') + 1:data_time.rfind('/')]
        data_time = year + "/" + month + "/" + day

    return data_time


def data_time_clean(data: object, data_time: object) -> object:
    """
    时间处理函数，时间列进行处理 形成 %Y%m%d 数据以及月份数据
    :param data:待处理数据 dataframe
    :param data_time:待处理数据中的日期列 日期格式包含日期
    :return:清洗好后的数据
    """
    #name = get_variable_name(data , namespace)
    logger.info(f"时间变量处理开始")
    data["clean_date"] = data[data_time].astype(str)
    data["clean_date"] = data["clean_date"].apply(data_time_change)
    data["clean_date"] = data["clean_date"].apply(lambda x: re.sub('[-/年月]', '', x))
    data["clean_date"] = data["clean_date"].str.strip()
    data["clean_date"]  = data["clean_date"].apply(
        lambda x: x + '01' if len(x) == 6 and x.isdigit() else x[:8] if len(x) >= 8 else x
    )
    data["clean_date"] = pd.to_datetime(data["clean_date"].str[0:8], format="%Y%m%d")
    data["clean_date"] = data["clean_date"].apply(lambda x: x.strftime('%Y%m%d'))
    data['clean_yearmonth'] = pd.to_datetime(data["clean_date"].str[0:8], format='%Y%m%d')
    data["clean_yearmonth"] = data["clean_yearmonth"].apply(lambda x: x.strftime('%Y%m'))
    logger.info(f"\t> 待处理时间的列为: {data_time}, 处理后时间列为: clean_date, 月份列为: clean_yearmonth")
    logger.info(f"时间变量处理结束")
    return data


####################################################################
#   数据清洗模块
####################################################################

# 缺失率处理
def calculate_missing_rate(data, cannot_delete_list, rate=0.8):
    """
    缺失率处理 删除缺失率高于 rate的值
    :param data: 待处理数据 dataframe
    :param cannot_delete_list:不能够删除的列
    :param rate:缺失率 0.8
    :return:处理后的数据 dataframe
    """
    # 数据集情况
    # rows, columns = data.shape
    logger.info("> 缺失率处理开始")
    # 计算每列的缺失率
    missing_rates = data.isnull().mean()
    # 找出缺失率高的列，可以根据实际情况设定一个阈值
    high_missing_rate_columns = [col for col, rate_eg in missing_rates.items() if rate_eg > rate]
    drop_columns = list(set(high_missing_rate_columns) - set(cannot_delete_list))
    processed_data = data.drop(columns=drop_columns)

    logger.info(f"\t> 原始数据集: 行数 {data.shape[0]}, 列数 {data.shape[1]}")
    logger.info(f"\t> 删除缺失率大于{rate:.1%}的值，共计删除 {data.shape[1] - processed_data.shape[1]} 列")
    logger.info(f"\t> 删除列为: {drop_columns}")
    # rows_r, columns_r = processed_data.shape
    logger.info(f"\t> 清洗数据集: 行数 {processed_data.shape[0]}, 列数 {processed_data.shape[1]}")
    logger.info("缺失率处理结束")
    return processed_data


# 同值率处理
def calculate_similarity_rate(column):
    """
    同值率计算
    :param column: 待计算dataframe列 df[]
    :return:similarity_rate 同值率
    """
    unique_values = column.unique()
    if len(unique_values) == 1:
        return 1.0
    else:
        most_common_value_count = column.value_counts().iloc[0]
        total_count = len(column)
        similarity_rate = most_common_value_count / total_count
        return similarity_rate

def nan_data_replace(data,Y_list):
    data.fillna(np.nan,inplace=True)
    data.replace('nan',np.nan, inplace=True)
    data.replace('/N', np.nan, inplace=True)
    for column in Y_list:
        data[column] = data[column].astype(int)
    return data

# 同值率数据处理
def drop_same_rate_high_column(data, cannot_delete_list, rate=0.8):
    """
    :param data: 待处理数据集
    :param cannot_delete_list:数据集中无需处理的列
    :param rate: 同值率的阈值
    :return: 清洗后的数据
    """
    logger.info("同值率处理开始")

    # 定义同值率集合
    similarity_rates = {}
    # 计算每列同值率
    total_iterations = len(data.columns) - 1
    for i, column_name in enumerate(data.columns):
        similarity_rates[column_name] = calculate_similarity_rate(data[column_name])
        log_progress(logger, i, total_iterations, prefix='\t> 处理中')
    high_similarity_columns = [column_name for column_name, value in similarity_rates.items() if value > rate]
    drop_columns = list(set(high_similarity_columns) - set(cannot_delete_list))
    processed_data = data.drop(columns=drop_columns)
    logger.info(f"\t> 原始数据集: 行数 {data.shape[0]}, 列数 {data.shape[1]}")
    logger.info(f"\t> 删除同值率大于{rate:.1%}的值, 共计删除 {data.shape[1] - processed_data.shape[1]} 列")
    logger.info(f"\t> 删除列为: {drop_columns}")
    logger.info(f"\t> 最终数据集为: 行数 {processed_data.shape[0]}, 列数 {processed_data.shape[1]}")
    logger.info("同值率处理完成")
    return processed_data

def monthly_y_distribution(df, target='y_flag', yearmonth_col='clean_yearmonth'):
    # 计算基础统计
    monthly = df.groupby(yearmonth_col)[target].agg(
        总样本数='count',
        坏样本数='sum',
        好样本数=lambda x: (x == 0).sum()
    ).reset_index()
    monthly['样本占比_数值'] = monthly['总样本数'] / monthly['总样本数'].sum()
    monthly['坏样本率_数值'] = monthly['坏样本数'] / monthly['总样本数']
    monthly['好样本率_数值'] = monthly['好样本数'] / monthly['总样本数']
    monthly['样本占比'] = monthly['样本占比_数值'].apply(lambda x: f'{x:.2%}')
    monthly['坏样本率'] = monthly['坏样本率_数值'].apply(lambda x: f'{x:.2%}')
    monthly['好样本率'] = monthly['好样本率_数值'].apply(lambda x: f'{x:.2%}')
    monthly['PSI'] = monthly[yearmonth_col].apply(
        lambda m: toad.metrics.PSI(
            df[df[yearmonth_col] == monthly[yearmonth_col].iloc[0]][target],
            df[df[yearmonth_col] == m][target]
        ) if m != monthly[yearmonth_col].iloc[0] else 0.0
    )
    monthly = monthly.drop(['样本占比_数值', '坏样本率_数值', '好样本率_数值'], axis=1)
    monthly = monthly.rename(columns={yearmonth_col: '年月'})
    return monthly

def under_sampling(data, labels, bad_sample_target):
    """
    欠采样/过采样（Oversampling）
        过采样是一种通过增加少数类样本数量来平衡数据集的方法。其主要思想是生成新的少数类样本，使少数类的样本数量增加到与多数类相同或相近。
    随机过采样
    data: 数据集 x
    labels 数据集 y
    bad_sample_target: 期望的坏样本占比
    """
    # 坏样本占比
    bad_proportion = len(np.where(labels == 1)[0]) / len(labels)
    # 坏样本数量
    bad_len = len(np.where(labels == 1)[0])
    # 好样本数量
    good_len = len(np.where(labels != 1)[0])
    # 抽样后好样本比例
    good_sample_target = 1 - bad_sample_target
    if bad_proportion < bad_sample_target:
        print("坏样本较少——过抽坏样本 随机抽样")
        # 总样本数
        total_target_num = round(good_len / good_sample_target)
        # 缺少的坏样本数
        need_bad = total_target_num - len(labels)
        print(" --过抽坏样本为：", need_bad)
        # 坏样本
        minority_class_indices = np.where(labels == 1)[0]
        # 固定随机数
        np.random.seed(42)

        # 针对坏数据随机过抽样
        oversampled_indices = np.random.choice(minority_class_indices,
                                               size=need_bad,
                                               replace=True)
        oversampled_data = data[oversampled_indices]
        oversampled_labels = labels[oversampled_indices]

        # 数据集为 array
        combined_data = np.vstack((data, oversampled_data))
        combined_target = np.hstack((labels, oversampled_labels))
    else:
        print("好样本较少——过抽好样本 随机抽样")
        # 样本总数
        total_target_num = round(bad_len / bad_sample_target)
        # 缺少好样本数量
        need_good = total_target_num - len(labels)
        print(" --过抽好样本为：", need_good)
        # 好样本
        minority_class_indices = np.where(labels != 1)[0]
        # 固定随机数
        np.random.seed(42)
        # 随机过抽样
        oversampled_indices = np.random.choice(minority_class_indices,
                                               size=need_good,
                                               replace=True)
        oversampled_data = data[oversampled_indices]
        oversampled_labels = labels[oversampled_indices]
        combined_data = np.vstack((data, oversampled_data))
        combined_target = np.hstack((labels, oversampled_labels))

    return combined_data, combined_target
