import lightgbm as lgb
from lightgbm import LGBMClassifier

from lightgbm import LGBMClassifier

def initialize_lgbm(space):
    """
    Initialize an LGBMClassifier using parameters from the `space` dictionary.
    If a parameter is not defined in `space`, the default parameter will be used.

    Parameters:
    space (dict): A dictionary containing hyperparameters for the LGBMClassifier.

    Returns:
    model (LGBMClassifier): An instance of the LGBMClassifier with the specified hyperparameters.
    """
    
    model = lgb.LGBMClassifier(**space)
    
    return model


if __name__ == "__main__":
    space = {
        # 'boosting_type':'gbdt',
        'num_leaves':31,
        'max_depth':3, 
        'learning_rate':0.02,
        'n_estimators':1000,
        # 'subsample_for_bin':200000,
        'objective':'binary',
        'class_weight':None,
        'min_split_gain':0.0,
        'min_child_weight':1e-3,
        'min_child_samples':500,
        'subsample':0.9,
        'subsample_freq':100,
        'colsample_bytree':0.9,
        'reg_alpha':0.0,
        'reg_lambda':0.0,
        'random_state':823,
        'n_jobs':-1,
        # 'importance_type':'split'
    }

    model = initialize_lgbm(space)
    print(dir(model))
    print(model)