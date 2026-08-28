import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectFromModel, RFE,RFECV
from src.utils.logger import setup_logging
logger = setup_logging('model_teain01')


# 控制模型训练入模变量
def model_data_clean(data,y_flag,NOT_MODEL_TRAIN_FEATURE):
    data[y_flag] = data[y_flag].astype(int)
    y_list = [0,1]
    model_data = data[data[y_flag].isin(y_list)]
    model_data_columns = model_data.columns.to_list()
    model_vars = []
    other_vars = []
    for var in model_data_columns:
        if var not in NOT_MODEL_TRAIN_FEATURE:
            model_vars.append(var)
        else:
            other_vars.append(var)
    logger.info(f"总变量数: {len(model_data_columns)}")
    logger.info(f"建模变量数: {len(model_vars)}")
    logger.info(f"其他变量数: {len(other_vars)}")
    logger.info(f"其他变量: {other_vars}")
    #logger.info(f"月份分布:{model_data['clean_yearmonth'].value_counts()}")
    return model_data,model_vars

# 好坏样本
def good_bad_info(data, flag):
    if data[flag].nunique() == 2:
        data[flag] = data[flag].astype(int)
        good = (data[flag] == 0).sum()
        bad = (data[flag]  == 1).sum()
        total = len(data[flag])
        return good, bad, total, round(bad / total, 2)
    else:
        logger.error("flag 中非两个值")


# 数据拆分
def sample_select(data, y_flag, vldt_ym=None):
    """
    :param data: 待分割数据集
    :param y_flag: 数据集中Y标签
    :param vldt_ym: 验证集的月份
    :return: 训练集特征、测试集特征、训练集标签、测试集标签、验证集特征、验证集标签、合并后的所有数据
    """

    test_size = 0.2
    random_state = 100
    data_info_columns = ["数据集", "行数", "列数", "好样本", "坏样本", "总样本数", "坏样本占比"]
    # 时间字段 因为使用时间清理方法所以yearmonth 默认为clean_yearmonth
    yearmonth = "clean_yearmonth"
    logger.info("开始数据分割")
    logger.info(f"\t> 分割参数: test_size: {test_size}, random_state: {random_state}")

    # 验证集数据集
    if vldt_ym:
        oot_data = True
    else:
        oot_data = False
    rows = []
    if oot_data:
        vldt_data = data[data[yearmonth].isin(vldt_ym)].copy()
        train_test_data = data[~data[yearmonth].isin(vldt_ym)]
        logger.info(f"\t> oot数据集为: {vldt_ym}")
        good, bad, total, bad_re = good_bad_info(vldt_data, y_flag)
        rows.append(["oot数据集", vldt_data.shape[0], vldt_data.shape[1], good, bad, total, bad_re])

    else:
        train_test_data = data

    # 训练测试数据拆分
    train_x, test_x, train_y, test_y = train_test_split(
        train_test_data.drop(y_flag, axis=1),
        train_test_data[y_flag],
        test_size=test_size,
        random_state=random_state,
        stratify=train_test_data[y_flag]
    )

    # 训练集数据
    train_data = pd.concat([train_x, train_y], axis=1)
    train_good, train_bad, train_total, train_bad_re = good_bad_info(train_data, y_flag)
    rows.append(["train数据集", train_data.shape[0], train_data.shape[1], train_good, train_bad, train_total, train_bad_re])

    # 测试集数据
    test_data = pd.concat([test_x, test_y], axis=1)
    test_good, test_bad, test_total, test_bad_re = good_bad_info(test_data, y_flag)
    rows.append(["test数据集", test_data.shape[0], test_data.shape[1], test_good, test_bad, test_total, test_bad_re])


    train_data.loc[:, 'samp_type'] = "train"
    test_data.loc[:, 'samp_type'] = "test"
    if oot_data:
        vldt_data.loc[:, 'samp_type'] = "vldt"
        all_data = pd.concat([train_data, test_data, vldt_data], axis=0)
    else:
        all_data = pd.concat([train_data, test_data], axis=0)
    logger.info("数据分割完成")

    data_info = pd.DataFrame(rows, columns=data_info_columns, index=None)
    return all_data, data_info




def model_vars_imp(model, method="all", num=100, rate=0.6):
    """
    根据指定的方法和标准筛选模型变量的重要性。

    参数:
    - models: 训练好的模型对象，用于提取特征重要性。
    - method: 字符串，指定筛选方法。可选值为 "all"（返回所有特征）、"auto"（自动根据累积重要性比例筛选）、"top"（返回顶部 num 个特征）。
    - num: 整数，当 method 为 "top" 时，指定返回的特征数量。
    - rate: 浮点数，当 method 为 "auto" 时，指定累积重要性比例。

    返回:
    - DataFrame，包含筛选后的模型变量名和对应的特征重要性。
    """
    # 创建DataFrame，包含模型变量名和对应的特征重要性，保持原始顺序
    df_imp = pd.DataFrame({
        'model_var': model.booster_.feature_name(),
        'var_imp': model.feature_importances_
    })

    # 根据num参数的值处理不同的情况
    if method == 'all':
        # 当num为all时，返回所有特征
        logger.info(f"变量的选择为：all")
        logger.info(f"模型变量总数为： {df_imp.shape[0]}")
        return df_imp['model_var'].to_list()

    elif method == "auto":
        # 当num为auto时，根据累积重要性比例自动选择特征
        logger.info(f"变量的选择为：auto")
        
        # 先按重要性降序排序用于累积计算，但保留原始索引
        df_imp_sorted = df_imp.sort_values(by='var_imp', ascending=False)
        total_importance = df_imp_sorted['var_imp'].sum()
        cumulative_importance = 0
        selected_count = 0
        # 遍历排序后的特征，直到累积重要性达到指定比例
        for _, row in df_imp_sorted.iterrows():
            cumulative_importance += row['var_imp']
            selected_count += 1
            if cumulative_importance / total_importance >= rate:
                break
        
        # 获取选中的特征名称
        selected_vars = df_imp_sorted.head(selected_count)['model_var'].tolist()
        
        # 按照原始顺序返回选中的特征
        result = df_imp[df_imp['model_var'].isin(selected_vars)]['model_var'].to_list()
        
        logger.info(f"模型变量筛选重要性占比{rate:.1%}的数为： {len(result)}")
        return result

    elif method == "top":
        # 当num为具体数字时，返回重要性最高的num个特征，但保持原始顺序
        if num < len(df_imp):
            logger.info(f"模型变量的选择为：TOP {num}")
            
            # 找到重要性最高的num个特征
            top_vars = df_imp.nlargest(num, 'var_imp')['model_var'].tolist()
            
            # 按照原始顺序返回这些特征
            result = df_imp[df_imp['model_var'].isin(top_vars)]['model_var'].to_list()
            
            logger.info(f"模型变量总数为： {len(result)}")
            return result
        else:
            logger.error("num值超过变量最大数")


def model_features_rfe_select(model, features, X_train, y_train, threshold="median",cv=3,scoring='roc_auc'):
    # 变量初筛选 保留变量重要性中位数以上的特征
    # mean 平均值，median 中位数，0.1*mean 保留重要性 ≥ 平均值 10% 的特征（需字符串格式）
    selector = SelectFromModel(model, threshold=threshold)  # 保留重要性中位数以上的特征
    selector.fit_transform(X_train, y_train)
    selected_mask = selector.get_support()

    # 初步筛选后的变量
    selected_feature_names = [features[i] for i, mask in enumerate(selected_mask) if mask]

    logger.info(f"SelectFromModel后剩余特征数: {len(selected_feature_names)}")
    logger.info(f"SelectFromModel后特征为: {selected_feature_names}")

    # 递归特征消除法 RFE
    #rfe = RFE(model)  # n_features_to_select=10选择前10个重要特征
    rfe = RFECV(
        estimator=model,  # 新模型实例
        cv=cv,  # 交叉验证折数
        scoring=scoring,  # 评估指标（默认用模型的score方法）
        step=1  # 每次移除1个特征（可调整为更大值加速）
    )

    rfe.fit_transform(X_train[selected_feature_names], y_train)

    rfe_mask = rfe.get_support()
    features = [selected_feature_names[i] for i, mask in enumerate(rfe_mask) if mask]

    #selected_features = X_train.columns[selector.get_support()][rfe.get_support()]
    #features = selected_features.tolist()
    logger.info(f"RFE后剩余特征数: {len(features)}")
    logger.info(f"RFE后最终特征为:{features}")

    return features
