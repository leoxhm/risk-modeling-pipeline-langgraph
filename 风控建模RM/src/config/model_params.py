LGB_PRPAMS = {           
    'bys': 
    {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_samples': [20, 50, 100, 150, 200],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 1, 5, 10, 20],
                'reg_lambda': [0, 1, 5, 10],
                'max_bin': [100, 150, 200, 250, 300]
            },
    'grid': 
    {
                'n_estimators': [100, 200, 300],
                #'learning_rate': [0.1],
                'max_depth': [3, 5]
               # 'subsample': [0.9]
            },
    'default':
    {
                'n_estimators': 100,
                #'learning_rate': [0.1],
                'max_depth': 3
    }
            }