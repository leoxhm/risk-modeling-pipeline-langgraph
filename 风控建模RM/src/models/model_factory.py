from src.models.lightgbm_model import lgb_model_train
from src.config.model_params import LGB_PRPAMS
from src.utils.logger import setup_logging
logger = setup_logging('model_train')

MODEL_TRAIN_TYPE = {
    "lgb": lgb_model_train
} 
MODEL_PARAMS_TYPE = {
    "lgb": LGB_PRPAMS
} 


def reorder_columns(df, exclude_col='model'):
    """重新排序列，将vldt_ks放在test_ks后面，vldt_auc放在test_auc后面"""
    cols = [c for c in df.columns if c != exclude_col]
    def get_col_order(col):
        if col == 'vldt_ks':
            return cols.index('test_ks') + 0.5 if 'test_ks' in cols else len(cols)
        elif col == 'vldt_auc':
            return cols.index('test_auc') + 0.5 if 'test_auc' in cols else len(cols)
        else:
            return cols.index(col)
    ordered_cols = sorted(cols, key=get_col_order)
    return df[ordered_cols + [exclude_col]]



def get_model_train(model_type):
    model_type = model_type.lower()
    if model_type not in MODEL_TRAIN_TYPE:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(MODEL_TRAIN_TYPE.keys())}")
    return MODEL_TRAIN_TYPE[model_type]


def model_train(model_data,
                 model_vars, 
                 y_flag, 
                 model_type,
                 model_tuning="default",     
                 model_params=None,
                 random_state=42):
    """
    :param y_flag:
    :param model_vars:
    :param model_data:
    :param model_type: 模型名称
    :param model_tuning: 模型调参
    :param model_params: 模型参数
    :return:
    """
    logger.info("开始模型训练")
    logger.info(f"\t> 模型名称: {model_type}")
    logger.info(f"\t> 模型调参方式: {model_tuning}")
    if model_params is None:
        model_params = MODEL_PARAMS_TYPE.get(model_type).get(model_tuning)
    logger.info(f"\t> 模型参数范围为: {model_params}")
    logger.info(f"\t> 入模变量数量为: {len(model_vars)}")
    model_train_info = get_model_train(model_type)(
        model_data=model_data,
        model_vars=model_vars,
        target=y_flag,
        method=model_tuning,
        random_state=random_state,
        params=model_params
    )
    logger.info("模型训练完成!")
    return model_train_info
