# !/usr/bin/env python
# -*- coding:utf-8 -*-

# 标准环境
import pandas as pd 
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler


# 自定义包
from src.models.model_tran_utlis import model_train_score
from src.utils.logger import setup_logging
logger = setup_logging('model_train',log_dir='./logs', log_name='model_train_info')


def model_info_print(model_info, params_combo):
    """打印模型训练信息"""
    
    logger.info(f"{'='*60}")
    logger.info(f"参数配置:")
    for key, value in params_combo.items():
        logger.info(f"   {key}: {value}")

    logger.info(f"模型表现:")
    
    def format_metric(value):
        """格式化指标值"""
        if value is None:
            return 'N/A'
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        if isinstance(value, str) and value == 'N/A':
            return 'N/A'
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)
    
    # 打印训练集指标
    train_ks = model_info.get('train_ks')
    train_auc = model_info.get('train_auc')
    if model_info.get('is_overfit'):
        logger.warning(f"   模型出现过拟合，惩罚得分: { model_info.get('final_score'):.4f}")
    logger.info(f"   训练集 KS: {format_metric(train_ks)}   |  AUC: {format_metric(train_auc)}")
    
    # 打印测试集指标
    test_ks = model_info.get('test_ks')
    test_auc = model_info.get('test_auc')
    logger.info(f"   测试集 KS: {format_metric(test_ks)}   |  AUC: {format_metric(test_auc)}")
    
    # 打印验证集指标
    vldt_ks = model_info.get('vldt_ks')
    if vldt_ks is not None:
        vldt_auc = model_info.get('vldt_auc')
        logger.info(f"   验证集 KS: {format_metric(vldt_ks)}   |  AUC: {format_metric(vldt_auc)}")

    # 打印PSI
    psi = model_info.get('train_test_psi')
    if psi is not None:
        logger.info(f"   train_test_psi: {format_metric(psi)}")

    if vldt_ks is not None:    
        train_vldt_psi = model_info.get('train_vldt_psi')
        if train_vldt_psi is not None:
            logger.info(f"   train_vldt_psi: {format_metric(train_vldt_psi)}")
        
        train_test_vldt_psi = model_info.get('train_test_vldt_psi')
        if train_test_vldt_psi is not None:
            logger.info(f"   train_test_vldt_psi: {format_metric(train_test_vldt_psi)}")
    
    logger.info(f"{'='*60}\n\n")

def model_init(params, random_state):
    """初始化LGB模型"""
    lgb_default_params = {
        'random_state': random_state,
        'n_jobs': -1,
        'verbose': -1
    }
    # 修复：合并两个字典而不是用set
    model_params = {**lgb_default_params, **params}
    model = LGBMClassifier(**model_params)
    return model


def grid_random_parameter_search(method,
                                 params,
                                 model_data,
                                 model_vars,
                                 target,
                                 random_state,
                                 n_iter):
    """
    网格搜索或随机搜索
    """
    # 生成参数组合
    if method =="grid":
        param_info = list(ParameterGrid(params))
        logger.info(f"网格搜索：共 {len(param_info)} 个参数组合")
    elif method == "random":
        param_info = list(ParameterSampler(params, n_iter=n_iter, random_state=random_state))
        logger.info(f"随机搜索：共 {len(param_info)} 个参数组合")

    # 存储所有训练结果
    model_train_result = []

    # 数据拆分
    train_data = model_data[model_data['samp_type'] == 'train']
    X_train = train_data[model_vars]
    y_train = train_data[target]

    # 模型训练
    for i, params_combo  in enumerate(param_info, 1):
        logger.info(f"正在训练第 {i}/{len(param_info)} 个模型，参数: {params_combo}")

        # 初始化并训练模型
        model = model_init(params_combo, random_state)
        model.fit(X_train, y_train)
        model_result_score = model_train_score(model_data,model,model_vars,target)
        model_info = {
            'model': model,
            'params': params_combo,
            "train_ks":model_result_score.get("trainKS"),
            "test_ks":model_result_score.get("testKS"),
            "train_auc":model_result_score.get("trainAUC"),
            "test_auc":model_result_score.get("testAUC"),
            "train_test_psi":model_result_score.get("train_test_psi"),
        }
        # 添加验证集结果
        if model_result_score.get("vldtKS") is not None:
            model_info['vldt_ks'] = model_result_score.get("vldtKS")
            model_info['vldt_auc'] = model_result_score.get("vldtAUC")
            model_info['train_vldt_psi'] = model_result_score.get("train_vldt_psi")
            model_info['train_test_vldt_psi'] = model_result_score.get("train_test_vldt_psi")
        model_train_result.append(model_info)
        model_info_print(model_info,params_combo)

    return model_train_result

def bayesian_parameter_search(params, model_data, model_vars, target, random_state, n_iter):
    """
    贝叶斯优化参数搜索 (基于 bayesian-optimization 的 BayesianOptimization)
    
    支持两种参数空间定义方式：
        - list: 离散选择，内部通过索引代理映射回真实值
        - tuple: 连续/整数范围，直接作为 pbounds 边界（元素必须是数值型）
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError:
        raise ImportError("贝叶斯优化需要安装 bayesian-optimization，请执行: pip install bayesian-optimization")
    
    # 数据拆分
    train_data = model_data[model_data['samp_type'] == 'train']
    X_train = train_data[model_vars]
    y_train = train_data[target]
    
    model_train_result = []
    
    # LightGBM 中通常需要取整的参数
    INT_PARAMS = {
        'num_leaves', 'max_depth', 'n_estimators', 'min_child_samples',
        'min_data_in_leaf', 'max_bin', 'bagging_freq', 'verbose',
        'random_state', 'n_jobs', 'min_child_weight', 'subsample_freq'
    }
    FLOAT_PRECISION = 4

    def clean_bayes_params(params_dict, int_params):
        """
        清洗贝叶斯优化参数:
        1. int参数 -> int
        2. float参数 -> round
        3. numpy类型 -> python原生类型
        """

        clean_params = {}
        for k, v in params_dict.items():
            # numpy类型转python原生类型
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            # int参数
            if k in int_params:
                clean_params[k] = int(round(v))
            # float参数
            elif isinstance(v, float):
                clean_params[k] = round(v, FLOAT_PRECISION)
            else:
                clean_params[k] = v
        return clean_params

    # 解析参数空间
    pbounds = {}
    discrete_map = {}   # key: 真实参数名, value: 可选列表
    fixed_params = {}   # 非搜索的固定参数
    
    for key, val in params.items():
        if isinstance(val, tuple) and len(val) == 2:
            # 强制转换为 float，避免字符串边界值导致 bayes_opt 内部崩溃
            try:
                low, high = float(val[0]), float(val[1])
                if low >= high:
                    raise ValueError(f"参数 {key} 的范围边界必须满足 low < high，当前: ({low}, {high})")
                pbounds[key] = (low, high)
            except (ValueError, TypeError) as e:
                # 如果转换失败（如传入的是 ('gbdt','dart')），当作离散参数处理
                logger.warning(f"参数 {key} 的值 {val} 无法作为数值范围解析，已转为离散选择")
                pbounds[f"{key}_idx"] = (0, len(val) - 1)
                discrete_map[key] = list(val)
        elif isinstance(val, list):
            if len(val) == 0:
                raise ValueError(f"参数 {key} 的列表不能为空")
            pbounds[f"{key}_idx"] = (0, len(val) - 1)
            discrete_map[key] = val
        else:
            fixed_params[key] = val
    
    if not pbounds:
        raise ValueError("贝叶斯优化至少需要1个可搜索的参数，请检查 params 配置")

    def _objective(**kwargs):
        """bayes_opt 目标函数：最大化 test_auc"""
        trial_params = fixed_params.copy()
        # 1) 连续参数
        for key in pbounds:
            if not key.endswith("_idx"):
                trial_params[key] = kwargs[key]
        # 2) 离散参数
        for key, choices in discrete_map.items():
            idx_key = f"{key}_idx"
            idx = int(round(kwargs[idx_key]))
            idx = max(0, min(idx, len(choices) - 1))
            trial_params[key] = choices[idx]
        # 3) 参数统一清洗
        trial_params = clean_bayes_params(
            trial_params,
            INT_PARAMS
        )
        try:
            model = model_init(trial_params, random_state)
            model.fit(X_train, y_train)
            model_result_score = model_train_score(
                model_data,
                model,
                model_vars,
                target
            )

            # 核心指标
            train_ks = model_result_score.get("trainKS")
            test_ks = model_result_score.get("testKS")
            vldt_ks = model_result_score.get("vldtKS")
            train_auc = model_result_score.get("trainAUC")
            test_auc = model_result_score.get("testAUC")
            # 目标指标（优先验证集，其次测试集）
            target_ks = vldt_ks if vldt_ks is not None else test_ks
            if target_ks is None:
                return 0.0
            target_ks = float(target_ks)
            # 风控建模过拟合检查
            is_overfit = False
            # 检查1：训练集KS显著高于验证集/测试集
            if train_ks is not None:
                train_ks = float(train_ks)
                if vldt_ks is not None:
                    if train_ks - float(vldt_ks) > 0.05:
                        is_overfit = True
                elif test_ks is not None:
                    if train_ks - float(test_ks) > 0.05:
                        is_overfit = True
            # 检查2：AUC过拟合
            if train_auc is not None and test_auc is not None:
                if float(train_auc) - float(test_auc) > 0.03:
                    is_overfit = True
            # 检查3：PSI过大（分布偏移）
            psi = model_result_score.get("train_test_psi")
            if psi is not None and float(psi) > 0.25:
                is_overfit = True
            # 过拟合模型直接惩罚
            if is_overfit:
                target_ks = target_ks * 0.5  # 严重惩罚过拟合模型
            # 非过拟合模型，给予稳定性奖励
            stability_bonus = 0.0
            if vldt_ks is not None and test_ks is not None:
                ks_stability = 1 - abs(float(vldt_ks) - float(test_ks))
                stability_bonus = ks_stability * 0.02
            if psi is not None:
                psi_bonus = max(0, (0.25 - float(psi)) / 0.25) * 0.01
                stability_bonus += psi_bonus
            final_score = target_ks + stability_bonus

            # 保存结果
            model_info = {
                'model': model,
                'params': trial_params.copy(),
                'final_score': final_score,
                'is_overfit': is_overfit,
                "train_ks": model_result_score.get("trainKS"),
                "test_ks": model_result_score.get("testKS"),
                "train_auc": model_result_score.get("trainAUC"),
                "test_auc": model_result_score.get("testAUC"),
                "train_test_psi": model_result_score.get("train_test_psi"),
            }
            # 验证集
            if model_result_score.get("vldtKS") is not None:
                model_info['vldt_ks'] = model_result_score.get("vldtKS")
                model_info['vldt_auc'] = model_result_score.get("vldtAUC")
                model_info['train_vldt_psi'] = model_result_score.get(
                    "train_vldt_psi"
                )
                model_info['train_test_vldt_psi'] = model_result_score.get(
                    "train_test_vldt_psi"
                )
            model_train_result.append(model_info)
            model_info_print(model_info, trial_params)
            return final_score
        except Exception as e:
            logger.error(
                f"贝叶斯优化 trial 失败，参数: {trial_params}，错误: {e}"
            )
            return -999
        
    logger.info(f"贝叶斯优化：共 {n_iter} 轮迭代，搜索空间: {list(pbounds.keys())}\n")
    
    # 添加计数器
    objective_call_count = [0]
    original_objective = _objective

    def wrapped_objective(**kwargs):
        objective_call_count[0] += 1
        logger.info(f"第 {objective_call_count[0]} 次贝叶斯目标优化")
        return original_objective(**kwargs)

    optimizer = BayesianOptimization(
        f=wrapped_objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=0
    )
    init_points = min(5, max(2, n_iter // 5))
    optimizer.maximize(init_points=init_points, n_iter=n_iter - init_points)

    # 按target降序排列
    model_train_result.sort(key=lambda x: x.get('final_score', -999), reverse=True)
    best_model = model_train_result[0]
    if model_train_result[0].get("vldt_ks") is not None:
        if optimizer.max and optimizer.max['target'] > 0:
            logger.info(f"贝叶斯优化完成，最优模型 [0]:")
            logger.info(f"\t> 目标最优得分: {best_model.get('final_score', 0):.4f}")
            logger.info(f"\t> 最终训练集 KS: {best_model.get('train_ks', 0)}")
            logger.info(f"\t> 最终测试集 KS: {best_model.get('test_ks', 0)}")
            logger.info(f"\t> 最终验证集 KS: {best_model.get('vldt_ks', 0)}")
    elif model_train_result[0].get("test_ks") is not None:
        if optimizer.max and optimizer.max['target'] > 0:
            logger.info(f"贝叶斯优化完成，最优模型 [0]:")
            logger.info(f"\t> 目标最优得分: {best_model.get('final_score', 0):.4f}")
            logger.info(f"\t> 最终训练集 KS: {best_model.get('train_ks', 0)}")
            logger.info(f"\t> 最终测试集 KS: {best_model.get('test_ks', 0)}")
    else:
        if optimizer.max and optimizer.max['target'] > 0:
            logger.info(f"贝叶斯优化完成，最优模型 [0]:")
            logger.info(f"\t> 目标最优得分: {best_model.get('final_score', 0):.4f}")
            logger.info(f"\t> 最终训练集 AUC: {best_model.get('train_auc', 0)}")
            logger.info(f"\t> 最终测试集 AUC: {best_model.get('test_auc', 0)}")

    return model_train_result


def default_parameter_search(params, model_data, model_vars, target, random_state):
        # 存储所有训练结果
    model_train_result = []

    # 数据拆分
    train_data = model_data[model_data['samp_type'] == 'train']
    X_train = train_data[model_vars]
    y_train = train_data[target]

    # 初始化并训练模型
    model = model_init(params, random_state)
    model.fit(X_train, y_train)
    model_result_score = model_train_score(model_data,model,model_vars,target)
    model_info = {
        'model': model,
        'params': params,
        "train_ks":model_result_score.get("trainKS"),
        "test_ks":model_result_score.get("testKS"),
        "train_auc":model_result_score.get("trainAUC"),
        "test_auc":model_result_score.get("testAUC"),
        "train_test_psi":model_result_score.get("train_test_psi"),
    }
    # 添加验证集结果
    if model_result_score.get("vldtKS") is not None:
        model_info['vldt_ks'] = model_result_score.get("vldtKS")
        model_info['vldt_auc'] = model_result_score.get("vldtAUC")
        model_info['train_vldt_psi'] = model_result_score.get("train_vldt_psi")
        model_info['train_test_vldt_psi'] = model_result_score.get("train_test_vldt_psi")
    model_train_result.append(model_info)
    model_info_print(model_info,params)
    return model_train_result



def lgb_model_train(model_data, model_vars, target, random_state, method, params,n_iter=10):
    """
    主函数：训练LGB模型并返回结果
    """

    # 参数搜索
    if method in ['grid','random']:
        search_result = grid_random_parameter_search(method, params,model_data ,model_vars, target, random_state,n_iter)
    elif method in ['default']:
        search_result = default_parameter_search(params,model_data ,model_vars, target, random_state)
    elif method =="bys":
        search_result = bayesian_parameter_search(params, model_data, model_vars, target, random_state, n_iter)

    return search_result