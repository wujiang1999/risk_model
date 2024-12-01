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
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 7,
    # 其他参数可以省略，这些会使用默认值
    }

    model = initialize_lgbm(space)
    print(dir(model))
    print(model.n_estimators)