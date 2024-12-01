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
    # lightgbm.LGBMClassifier(boosting_type='gbdt', 
    #                         num_leaves=31, 
    #                         max_depth=-1, 
    #                         learning_rate=0.1, 
    #                         n_estimators=100, 
    #                         subsample_for_bin=200000, 
    #                         objective=None, 
    #                         class_weight=None, 
    #                         min_split_gain=0.0, 
    #                         min_child_weight=0.001, 
    #                         min_child_samples=20, 
    #                         subsample=1.0, 
    #                         subsample_freq=0, 
    #                         colsample_bytree=1.0, 
    #                         reg_alpha=0.0, 
    #                         reg_lambda=0.0, 
    #                         random_state=None, 
    #                         n_jobs=None, 
    #                         importance_type='split', 
    #                         **kwargs)
    space = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 7,
    }

    model = initialize_lgbm(space)
    print(dir(model))
    print(model.n_estimators)