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

### 基础指标的计算 auc ks lift psi
### 给出基于sklearn-api及手撕版本
### 定义静态类 其中方法可以通过BasicIndexCal.ks_cal()直接调用
class BasicIndexCal:
    @staticmethod
    def preprocess_cal(label, pred, exclude_value = [-1, -99, -999, -9999]):
        label, pred = np.array(label), np.array(pred)
        label = np.where((label == -1) | (label == 3), np.nan, np.where(label == 2, 0, label))
        pred = np.where(np.isin(pred, exclude_value), np.nan, pred)
        valid_idx = np.where((~np.isnan(label)) & (~np.isnan(pred)))[0]
        return label[valid_idx], pred[valid_idx]
    
    @staticmethod
    def ks_cal(label, pred):
        label, pred = BasicIndexCal.preprocess_cal(label, pred)
        if np.corrcoef(label, pred)[0, 1] > 0:
            fpr, tpr, _ = roc_curve(label, pred)
        else:
            fpr, tpr, _ = roc_curve(label, 1 - pred)
        return np.max(tpr - fpr)
    
    @staticmethod
    def ks_cal_manual(label, pred):
        label, pred = BasicIndexCal.preprocess_cal(label, pred)
        if np.corrcoef(label, pred)[0, 1] > 0:
            data = pd.DataFrame({'label': label, 'pred': pred}).sort_values(['pred'], ascending = False).reset_index(drop = True)
        else:
            data = pd.DataFrame({'label': label, 'pred': 1 - pred}).sort_values(['pred'], ascending = False).reset_index(drop = True)
        data['cum_pos'] = (data['label'] == 1).cumsum()
        data['cum_neg'] = (data['label'] == 0).cumsum()
        data['tpr'] = data['cum_pos'] / (data['label'] == 1).sum()
        data['fpr'] = data['cum_neg'] / (data['label'] == 0).sum()
        return np.max(data['tpr'] - data['fpr'])
    
    ### AUC的计算涉及到数值积分 np里面的梯形积分 在此不给出手撕版本
    @staticmethod
    def auc_cal(label, pred):
        label, pred = BasicIndexCal.preprocess_cal(label, pred)
        return max(roc_auc_score(label, pred), roc_auc_score(label, 1 - pred))
    
    ### IV的计算 手撕版本可见feature_stats_by_bin 可以对于唯一值分箱 也可以等频分箱
    @staticmethod
    def iv_cal(label, pred, return_sub = False):
        label, pred = BasicIndexCal.preprocess_cal(label, pred)
        return toad.stats.IV(pred, label, return_sub = return_sub)
    
    ### 基于分位点计算头尾部lift
    @staticmethod
    def lift_cal(label, pred, quantile = 0.1):
        label, pred = BasicIndexCal.preprocess_cal(label, pred)
        top_q_percent_idx = np.argsort(pred)[-int(quantile * len(pred)):]
        bot_q_percent_idx = np.argsort(pred)[:int(quantile * len(pred))]
        top_q_percent_lift = np.mean(label[top_q_percent_idx]) / np.mean(label)
        bot_q_percent_lift = np.mean(label[bot_q_percent_idx]) / np.mean(label)
        return bot_q_percent_lift, top_q_percent_lift
    
        
        

if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.5, 0.3, 0.85]
    y_score1 = [(1 - x) for x in y_score]
    # print(np.corrcoef(y_true, y_score)[0, 1], np.corrcoef(y_true, y_score1)[0, 1])
    # print(BasicIndexCal.ks_cal(y_true, y_score), BasicIndexCal.ks_cal(y_true, y_score1))
    # print(BasicIndexCal.ks_cal_manual(y_true, y_score), BasicIndexCal.ks_cal_manual(y_true, y_score1))
    print(BasicIndexCal.lift_cal(y_true, y_score))


