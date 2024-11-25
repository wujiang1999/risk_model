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
import psutil

### 获取前一日ds
def bizdate():
    return str((datetime.datetime.now() - datetime.timedelta(days = 1)).strftime('%Y%m%d'))

### 将空列表转化为None
def return_list_or_none(input_list):
    if not input_list:  # 检查列表是否为空
        return None
    else:
        return input_list

### dp读取预先进行数据类型转化
def cast_to_float(input_list):
    if isinstance(input_list, str):
        input_list = [input_list]
    return ',\n'.join([f'cast({x} as float)' for x in input_list])

### 同时merge多个df
### reduce被用于对序列中的元素进行连续的二元操作
def merge_df(df_list, share_cols, merge_way = 'left'):
    return reduce(lambda left, right:pd.merge(left, right, on = share_cols, how = merge_way), df_list)

### 读取or存储pkl文件
def read_and_save_pickle(path, read = True, data = None):
    if read:
        with open(path, 'rb') as f:
            output = pickle.load(f)
        return output
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)

### 模型打分 涉及到多种形式存储的模型文件此处只包含sklearn接口的
def score_df(data, model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if isinstance(model, lgb.LGBMClassifier):
        score_list = model.predict_proba(data[model.feature_name_])[:, 1]
    if isinstance(model, xgb.sklearn.XGBClassifier):
        score_list = model.predict_proba(data[model._Booster.feature_names])[:, 1]
    return score_list

### 获取指定路径下所有文件的列表
def get_file(path):
    entries = os.listdir(path)
    files = [os.path.join(path, entry) for entry in entries if os.path.isfile(os.path.join(path, entry))]
    return sorted(files)

### 时间列的标准化
def df_time_process(data, time_col, add_month_col = False, month_col = 'month'):
    data[time_col] = pd.to_datetime(data[time_col])
    if add_month_col:
        data[month_col] = data[time_col].dt.strftime('%Y-%m')
    data[time_col] = data[time_col].dt.strftime('%Y-%m-%d')
    return data

### K折划分
def kfold_split(data, time_col, time_split, flag_col, kfold, shuffle = False):
    skf = StratifiedKFold(n_splits = kfold, shuffle = shuffle)
    data = df_time_process(data = data, time_col = time_col, add_month_col = False, month_col = 'month')
    data1 = data[data[time_col] <= time_split]
    data2 = data[data[time_col] > time_split]
    data1 ['kfold'] = -1
    data1 = data1.reset_index(drop = True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(data1, data1[flag_col])):
        data1.loc[val_idx, 'kfold'] = str(fold + 1)
    data2['kfold'] = 'oot'
    return pd.concat([data1, data2]).reset_index(drop = True)

### k折自动生成每折文件名 不常用函数
def generate_new_paths(pkl_path, items):
    # 获取文件夹路径和文件名
    dir_path, filename = os.path.split(pkl_path)
    # 分离文件名和扩展名
    base_name, ext = os.path.splitext(filename)
    # 为每个元素生成新的路径
    new_paths = os.path.join(dir_path, f"{base_name}_{items}{ext}")
    return new_paths

if __name__ == "__main__":
    print(bizdate())
    
    input_list = None
    print(return_list_or_none(input_list))
    
    input_list = ['a','b']
    print(cast_to_float(input_list))
    
    # # 获取当前进程的内存信息
    # process = psutil.Process()
    # mem_info = process.memory_info()

    # print(f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB")  # 常驻内存
    # print(f"VMS: {mem_info.vms / 1024 ** 2:.2f} MB")  # 虚拟内存
