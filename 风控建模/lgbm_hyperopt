import pandas as pd
import numpy as np
import toad
import warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)
import gc
gc.enable()
import time
import pickle
np.random.seed(823)
import logging
import os

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

### 注意 虽然该类中有特征及Y预处理功能 但是建议将预处理完的数据直接导入
### 还可以修改的地方 将space作为参数进行初始化-可选 调整subsample_freq 自定义损失函数 调整目标函数-当前是oos上auc 设置早停使用的指标等等
class LightGBMHyperoptModeling(BasicIndexCal):
    def __init__(self, ins, oos, oot, time_col, label_col, num_col, cate_col,
                 params_searching_path, model_path, model_score_path, log_path,
                 to_replace = [-1, -99, -999, -9999], 
                 stopping_rounds = 30, period = 100, alpha = 25, max_evals = 100, 
                 need_preprocess = False, drop_fea_col = False):
        """
        ins : DataFrame, train dataset
        oos : DataFrame, validation dataset
        oot : DataFrame, test dataset 
        time_col : str, time column name
        label_col : str, label column name
        num_col : list, numerical feature column names
        cate_col : list, categorical feature column names (default = None)
        params_searching_path : str, path to save params searching results by csv
        model_path : str, path to save model by pickle (default = None)
        model_score_path : str, path to save model scores by feather (default = None)
        log_path : str, path to save log by log (default = None)
        exclude_value : list, to_replace values in feature columns
        stopping_rounds : int, early stopping rounds
        period : int, evaluation period
        alpha : float, penalty factor for controlling overfitting
        max_evals : int, parameter searching rounds
        need_preprocess : bool, whether to preprocess data (default = False)
        drop_fea_col : bool, whether to drop feature columns (default = False)
        """
        self.ins = ins
        self.oos = oos
        self.oot = oot
        self.time_col = time_col
        self.label_col = label_col
        self.num_col = num_col
        self.cate_col = cate_col
        if not self.cate_col:
            self.fea_col = self.num_col + self.cate_col
        else:
            self.fea_col = self.num_col
        
        self.params_searching_path = params_searching_path
        self.model_path = model_path
        self.model_score_path = model_score_path
        self.log_path = log_path
        
        self.to_replace = to_replace
        self.stopping_rounds = stopping_rounds
        self.period = period
        self.alpha = alpha
        self.max_evals = max_evals
        self.need_preprocess = need_preprocess
        self.drop_fea_col = drop_fea_col
        self.results = []
        
        print('feature columns:%s len:%s'%(self.fea_col, len(self.fea_col)))
        print('INS SHAPE:', self.ins.shape)
        print('OOS SHAPE:', self.oos.shape)
        print('OOT SHAPE:', self.oot.shape)
        
        if need_preprocess:
            self.preprocess_data()
        if log_path:
            self.create_logger()
            
    def create_logger(self):
        self.log_dir = os.path.dirname(self.log_path)
        os.makedirs(self.log_dir, exist_ok = True)
        
        self.logger = logging.getLogger(self.log_path)
        self.logger.setLevel(logging.INFO)
    
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setLevel(logging.INFO)
    
        self.formatter = logging.Formatter('%(asctime)s - %(pathname)s - [%(levelname)s] - %(message)s')
        self.file_handler.setFormatter(self.formatter)
    
        if not self.logger.handlers:
            self.logger.addHandler(self.file_handler) 
        return self.logger
    
    def preprocess_data(self):
        def process_dataset(data):
            data[self.time_col] = pd.to_datetime(data[self.time_col])
            data[self.time_col] = data[self.time_col].dt.strftime('%Y-%m-%d')
            data[self.label_col] = pd.to_numeric(data[self.label_col], errors = 'coerce')
            data = data[data[self.label_col].isin([0, 1, 2])]
            data[self.label_col] = data[self.label_col].replace(2, 0)
            if not self.cate_col:
                for cc in self.cate_col:
                    data[cc], _ = pd.factorize(data[cc])
                    data[cc] = data[cc].astype('category')
            for nc in self.num_col:
                data[nc] = pd.to_numeric(data[nc], errors = 'coerce')
                data[nc] = data[nc].replace(self.exclude_value, np.nan)
            return data
        
        self.ins = process_dataset(self.ins)
        self.oos = process_dataset(self.oos)
        self.oot = process_dataset(self.oot)
    
    def hyperparameter_tuning(self, space):
        clf = lgb.LGBMClassifier(
            max_depth = int(space['max_depth']),
            learning_rate = space['learning_rate'],
            n_estimators = int(space['n_estimators']),
            class_weight = 'balanced',
            subsample_freq = 10,
            subsample = space['subsample'],
            colsample_bytree = space['colsample_bytree'],
            reg_alpha = space['reg_alpha'],
            reg_lambda = space['reg_lambda'],
            min_child_samples = int(space['min_child_samples']),
            min_split_gain = space['min_split_gain'],
            random_state = 823, 
            importance_type = 'gain',
            verbosity = -1
        )
        
        evaluation = [(self.oos[self.fea_col], self.oos[self.label_col])]
        early_stopping_callback = lgb.early_stopping(stopping_rounds = self.stopping_rounds, min_delta = 1e-4)
        log_evaluation_callback = lgb.log_evaluation(period = self.period)
        
        clf.fit(self.ins[self.fea_col], self.ins[self.label_col], eval_set = evaluation, categorical_feature = self.cate_col, callbacks = [early_stopping_callback, log_evaluation_callback], eval_metric = 'auc')
        
        pred_ins = clf.predict_proba(self.ins[self.fea_col])[:, 1]  
        auc_ins = self.auc_cal(self.ins[self.label_col], pred_ins)
        ks_ins = self.ks_cal(self.ins[self.label_col], pred_ins)
        lift_ins_head, lift_ins_tail = self.lift_cal(self.ins[self.label_col], pred_ins)
        print("INS AUC=%.3f , INS KS=%.3f , INS HEAD LIFT=%.3f , INS TAIL LIFT=%.3f" %(auc_ins, ks_ins, lift_ins_head, lift_ins_tail))
        
        pred_oos = clf.predict_proba(self.oos[self.fea_col])[:, 1]  
        auc_oos = self.auc_cal(self.oos[self.label_col], pred_oos)
        ks_oos = self.ks_cal(self.oos[self.label_col], pred_oos)
        lift_oos_head, lift_oos_tail = self.lift_cal(self.oos[self.label_col], pred_oos)
        print("OOS AUC=%.3f , OOS KS=%.3f , OOS HEAD LIFT=%.3f , OOS TAIL LIFT=%.3f" %(auc_oos, ks_oos, lift_oos_head, lift_oos_tail))
        
        pred_oot = clf.predict_proba(self.oot[self.fea_col])[:, 1]  
        auc_oot = self.auc_cal(self.oot[self.label_col], pred_oot)
        ks_oot = self.ks_cal(self.oot[self.label_col], pred_oot)
        lift_oot_head, lift_oot_tail = self.lift_cal(self.oot[self.label_col], pred_oot)
        print("OOT AUC=%.3f , OOT KS=%.3f , OOT HEAD LIFT=%.3f , OOT TAIL LIFT=%.3f" %(auc_oot, ks_oot, lift_oot_head, lift_oot_tail))
        
        result = space.copy()
        result.update(
                {'early_stopping_rounds':clf.best_iteration_ + 1,
                 'auc_train':auc_ins,
                 'auc_valid':auc_oos,
                 'auc_test':auc_oot,
                 'auc_diff1':auc_oos-auc_ins,
                 'auc_diff2':auc_oot-auc_oos,
                 'ks_train':ks_ins,
                 'ks_valid':ks_oos,
                 'ks_test':ks_oot,
                 'ks_diff1':ks_oos-ks_ins,
                 'ks_diff2':ks_oot-ks_oos,
                 'ins_head_lift':lift_ins_head,
                 'ins_tail_lift':lift_ins_tail,
                 'oos_head_lift':lift_oos_head,
                 'oos_tail_lift':lift_oos_tail,
                 'oot_head_lift':lift_oot_head,
                 'oot_tail_lift':lift_oot_tail,}
        )
        
        self.results.append(result)
        results_df = pd.DataFrame(self.results)
        if self.params_searching_path:
            results_df.to_csv(self.params_searching_path)

        if self.log_path:
            self.logger.info(f"Iteration results:{list(result.items())}")
            # print('log writing')
        
        if auc_oos >= auc_ins:
            loss = -auc_oos
        else:
            loss = -(auc_oos - self.alpha * (auc_ins - auc_oos)**2)
        
        return {'loss':loss, 'status':STATUS_OK}
    
    def optimize(self):
        space = {
            'max_depth':hp.quniform("max_depth", 2, 5, 1),
            'learning_rate':hp.quniform("learning_rate", 0.01, 0.2, 0.01),
            'n_estimators': hp.quniform('n_estimators', 500, 1200, 50),
            'subsample':hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha':hp.uniform('reg_alpha', 0, 100),
            'reg_lambda':hp.uniform('reg_lambda', 0, 100),
            'min_child_samples':hp.quniform('min_child_samples', 5000, 50000, 1000),
            'min_split_gain':hp.uniform('min_split_gain', 0, 100),
        }
        
        trials = Trials()
        best = fmin(fn = self.hyperparameter_tuning, 
                  space = space, 
                  algo = tpe.suggest,
                  max_evals = self.max_evals,
                  trials = trials)
        
        return best
    
    def fit(self):
        start = time.time()
        
        best_params = self.optimize()
        
        if self.model_path:
            model = lgb.LGBMClassifier(
                max_depth = int(best_params['max_depth']),
                learning_rate = best_params['learning_rate'],
                n_estimators = int(best_params['n_estimators']),
                class_weight = 'balanced',
                subsample_freq = 10,
                subsample = best_params['subsample'],
                colsample_bytree = best_params['colsample_bytree'],
                reg_alpha = best_params['reg_alpha'],
                reg_lambda = best_params['reg_lambda'],
                min_child_samples = int(best_params['min_child_samples']),
                min_split_gain = best_params['min_split_gain'],
                random_state = 823, 
                importance_type = 'gain',
                verbosity = -1
            )

            evaluation = [(self.oos[self.fea_col], self.oos[self.label_col])]
            early_stopping_callback = lgb.early_stopping(stopping_rounds = self.stopping_rounds, min_delta = 1e-4)
            log_evaluation_callback = lgb.log_evaluation(period = self.period)

            model.fit(self.ins[self.fea_col], self.ins[self.label_col],
                    eval_set = evaluation, categorical_feature = self.cate_col,
                    callbacks = [early_stopping_callback, log_evaluation_callback], eval_metric = 'auc')

            with open(self.model_path, 'wb') as file:
                pickle.dump(model, file)
            
            if self.model_score_path:
                self.ins['proba'] = model.predict_proba(self.ins[self.fea_col])[:, 1]
                self.oos['proba'] = model.predict_proba(self.oos[self.fea_col])[:, 1]
                self.oot['proba'] = model.predict_proba(self.oot[self.fea_col])[:, 1]
                output = pd.concat([self.ins, self.oos, self.oot])
                if self.drop_fea_col:
                    output = output.drop(self.fea_col, axis = 1)
                output = output.reset_index(drop = True)
                output.to_feather(self.model_score_path)
        
        total_time = (time.time() - start) / 60
        print('Finished')
        print('Total time taken %.2f mins' %total_time)
        
if __name__ == "__main__":
    print(823)