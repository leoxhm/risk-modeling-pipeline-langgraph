import pandas as pd 
import numpy as np
from toad.metrics import KS,PSI,AUC

def model_train_score(model_data,model,model_vars,y_flag):

    # 初始化存储模型评分结果的字典
    model_result_score = {}
    psi_gap = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    type_list = model_data['samp_type'].unique().tolist()

    # 打分
    model_data['model_pred'] = model.predict_proba(model_data[model_vars])[:, 1]

    # 提前缓存各类型数据以减少重复计算
    data_by_type = {
        samp_type: model_data[model_data['samp_type'] == samp_type]
        for samp_type in type_list
    }

    for samp_type in type_list:
        data = data_by_type[samp_type]
        y = data[y_flag]
        y_pred = data['model_pred']
        ks = KS(y_pred, y)
        auc = AUC(y_pred, y)
        model_result_score[samp_type + "KS"] = f"{ks:.3f}"
        model_result_score[samp_type + "AUC"] = f"{auc:.3f}"

    # 获取训练集和测试集用于 PSI
    train_pred = data_by_type.get('train', None)['model_pred'] if 'train' in data_by_type else None
    test_pred = data_by_type.get('test', None)['model_pred'] if 'test' in data_by_type else None

    if train_pred is not None and test_pred is not None:
        train_test_psi = PSI(train_pred, test_pred, psi_gap)
        model_result_score["train_test_psi"] = f"{train_test_psi:.4f}"
   
    if "vldt" in type_list:
        vldt_pred = data_by_type['vldt']['model_pred']
        
        if train_pred is not None:
            train_vldt_psi = PSI(train_pred, vldt_pred, psi_gap)
            model_result_score["train_vldt_psi"] = f"{train_vldt_psi:.4f}"

        if train_pred is not None and test_pred is not None:
            train_test_vldt_psi = PSI(
                pd.concat([train_pred, test_pred]), vldt_pred, psi_gap
            )
            model_result_score["train_test_vldt_psi"] = f"{train_test_vldt_psi:.4f}"
    else:
        model_result_score['vldtKS'] = None

    return model_result_score



