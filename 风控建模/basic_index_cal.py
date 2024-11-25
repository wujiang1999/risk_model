import math
import random
import pandas as pd
import numpy as np
import toad
import warnings
import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pd.options.mode.chained_assignment = None
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)
import gc
gc.enable()
import time
import pickle
from functools import reduce
from sklearn.model_selection import StratifiedKFold
np.random.seed(823)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import logging
import os

### 对于下列所有函数，都可以使用toad原生库进行计算
### toad计算分箱时，通常是每一个唯一值分一箱，为了减少计算量可以预分箱
### 特征分箱并计算woe、iv、ks等基础指标
def feature_evaluate_by_bin(df, feature, label, bin_nums = 10, precision = 3):
    data = df.copy()
    data[feature] = data[feature].replace(np.nan, -9999)
    data[label] = pd.to_numeric(data[label])
    data = data[data[label].isin([0, 1, 2])]
    data[label] = data[label].apply(lambda x:1 if x == 1 else 0)
    
    ### 有些特征天然有负值，需要排除这种情况
    ### ==可能存在数据精度问题
    if_empty = (data[feature].min() == -9999)
    unique_values_count = data[feature].nunique()
    if unique_values_count <= 10:
        if if_empty:
            less_than_zero = pd.cut(data[feature][data[feature] == -9999], bins = 1, labels = ['unmatched'])
            more_than_zero = pd.qcut(data[feature][data[feature] > -9999], q = unique_values_count, duplicates = 'drop')
            data_cut = pd.concat([more_than_zero, less_than_zero])
        else:
            data_cut = pd.qcut(data[feature][data[feature] > -9999], q = unique_values_count, duplicates = 'drop')
    else:
        if if_empty:
            less_than_zero = pd.cut(data[feature][data[feature] == -9999], bins = 1, labels = ['unmatched'])
            more_than_zero = pd.qcut(data[feature][data[feature] > -9999], q = bin_nums, duplicates = 'drop')
            data_cut = pd.concat([more_than_zero, less_than_zero])
        else:
            data_cut = pd.qcut(data[feature][data[feature] > -9999], q = bin_nums, duplicates = 'drop')
    
    cut_group_all = data[feature].groupby(data_cut).count()
    ### 统计正样本及负样本数量
    cut_y = data[label].groupby(data_cut).sum()
    cut_n = cut_group_all - cut_y    
    tp = pd.DataFrame()
    tp['total'] = cut_group_all
    tp['bad_count'] = cut_y
    tp['good_count'] = cut_n
    tp['percent'] = tp['total'] / tp['total'].sum()
    
    tp['good_rate'] = (tp['good_count'] + 1e-6) / tp['good_count'].sum()  
    tp['bad_rate'] = (tp['bad_count'] + 1e-6)  / tp['bad_count'].sum()
    tp['woe'] = np.log(tp['good_rate'] / tp['bad_rate'])
    tp['ivn'] = tp['woe'] * (tp['good_rate'] - tp['bad_rate'])
    tp['iv'] = tp['ivn'].sum()
    
    tp['ksn'] = (tp['good_rate'].cumsum() - tp['bad_rate'].cumsum()).abs()
    tp['ks'] = tp['ksn'].max()
    
    tp['badrate'] = tp['bad_count'] / tp['total']
    tp['lift'] = tp['badrate'] / (tp['bad_count'].sum() / tp['total'].sum())
    tp['bad_count'] = tp['bad_count'].astype(int)
    tp['good_count'] = tp['good_count'].astype(int)
    ### 对于dataframe中的字段标准化处理
    tp['percent'] = tp['percent'].map(lambda x: f"{x:.2%}")
    tp['good_rate'] = tp['good_rate'].map(lambda x: f"{x:.2%}")
    tp['bad_rate'] = tp['bad_rate'].map(lambda x: f"{x:.2%}")
    tp['badrate'] = tp['badrate'].map(lambda x: f"{x:.2%}")
    tp['ksn'] = tp['ksn'].round(precision)
    tp['ks'] = tp['ks'].round(precision)
    tp['woe'] = tp['woe'].round(precision)
    tp['ivn'] = tp['ivn'].round(precision)
    tp['iv'] = tp['iv'].round(precision)
    tp['lift'] = tp['lift'].round(precision)
    return tp



if __name__ == "__main__":
    feature_column = np.random.uniform(0, 100, 1000)
    label_column = np.random.randint(0, 2, 1000)
    # 创建 DataFrame
    df = pd.DataFrame({
        'feature': feature_column,
        'label': label_column
    })
    print(feature_evaluate_by_bin(df, 'feature', 'label'))
    
    print(-9999 == -9999)