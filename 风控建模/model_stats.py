import pandas as pd
import numpy as np
import toad
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
pd.options.mode.chained_assignment = None
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)
import gc
gc.enable()
np.random.seed(823)
from code_tool import df_time_process

### 基础指标的计算 auc ks lift psi 给出基于sklearn-api及手撕版本
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
    
### 基础指标矩阵计算
class BasicMatrixCal(BasicIndexCal):
    def __init__(self, data, time_col, fea_col, label_col, precision = 4):
        self.data = data
        self.time_col = time_col
        self.fea_col = fea_col
        if not label_col:
            self.label_col = label_col
        self.precision = precision
        
        self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
        self.data['Month'] = self.data[self.time_col].dt.strftime('%Y-%m')
        self.month_list = self.data['Month'].unique()

    def ks_matrix_cal(self):
        ks_matrix = pd.DataFrame(np.zeros((len(self.month_list), len(sorted(self.label_col)))), index = sorted(self.month_list), columns = sorted(self.label_col))
        for i in ks_matrix.index:
            for j in ks_matrix.columns:
                data_slice = self.data[self.data['Month'] == i]
                try:
                    ks_matrix.loc[i, j] = self.ks_cal(data_slice[j], data_slice[self.fea_col])
                except:
                    ks_matrix.loc[i, j] = np.nan
        return round(ks_matrix, self.precision)

    def auc_matrix_cal(self):
        auc_matrix = pd.DataFrame(np.zeros((len(self.month_list), len(sorted(self.label_col)))), index = sorted(self.month_list), columns = sorted(self.label_col))
        for i in auc_matrix.index:
            for j in auc_matrix.columns:
                data_slice = self.data[self.data['Month'] == i]
                try:
                    auc_matrix.loc[i, j] = self.auc_cal(data_slice[j], data_slice[self.fea_col])
                except:
                    auc_matrix.loc[i, j] = np.nan
        return round(auc_matrix, self.precision)
    
### 特征按月分箱观察变量稳定性和偏移情况
def bin_by_month(df, time_col, fea_col, bin_nums = 10, base_month = False, epsilon = 1e-6,if_format = True):
    data = df.copy()
    data[time_col] = pd.to_datetime(df[time_col])
    data['month'] = data[time_col].dt.strftime('%Y-%m')
    mon_list = list(sorted(data['month'].unique()))
    if base_month:
        _, bins = pd.qcut(data[data['month'].isin(list(base_month))][fea_col], q = bin_nums, retbins = True)
    else:
        _, bins = pd.qcut(data[data['month'] == mon_list[0]][fea_col], q = bin_nums, retbins = True)
    bins[0], bins[-1] = -np.inf, np.inf
    bins = np.unique(bins)
    
    fenxianglist = []
    for mon in mon_list:
        data_slice = data[data['month'] == mon].copy()
        data_slice['bin'] = pd.cut(data_slice[fea_col], bins = bins, include_lowest = True)
        fenxiang_by_month = data_slice.groupby('bin', dropna = False).agg({'month':'count'}) / len(data_slice)
        fenxiang_by_month.columns = [mon]
        fenxianglist.append(fenxiang_by_month)
        # if if_format:
        #     fenxiang_by_month[mon] = fenxiang_by_month[mon].map(lambda x:'{:.2%}'.format(x))
    output = pd.concat(fenxianglist, axis = 1)
    psi_matrix = pd.DataFrame(np.zeros((len(mon_list), len(mon_list))), index = mon_list, columns = mon_list)
    for i in range(len(mon_list)):
        for j in range(len(mon_list)):
            psi_matrix.iloc[i, j] = np.sum((output.iloc[:, i] - output.iloc[:, j]) * (np.log(output.iloc[:, i] + epsilon) - np.log(output.iloc[:, j] + epsilon)))
    if if_format:
        for col in output.columns:
            output[col] = output[col].map(lambda x:'{:.2%}'.format(x))
    return output, psi_matrix
    
        
if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.5, 0.3, 0.85]
    y_score1 = [(1 - x) for x in y_score]
    # print(np.corrcoef(y_true, y_score)[0, 1], np.corrcoef(y_true, y_score1)[0, 1])
    # print(BasicIndexCal.ks_cal(y_true, y_score), BasicIndexCal.ks_cal(y_true, y_score1))
    # print(BasicIndexCal.ks_cal_manual(y_true, y_score), BasicIndexCal.ks_cal_manual(y_true, y_score1))
    print(BasicIndexCal.lift_cal(y_true, y_score))


