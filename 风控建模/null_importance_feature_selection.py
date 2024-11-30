import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gc
gc.enable()
import time


### 通过shuffle y来计算实际特征重要性
class NullImportanceFeatureSelection():
    def __init__(self, data, time_col, label_col,  num_col, cate_col,
                 actual_imp_df_path, null_imp_df_path, scores_df_path,
                 need_preprocess = False, shuffle = False, seed = 823, 
                 num_boost_round = 200, rounds = 100):
        '''
        data : DataFrame
        time_col : str, time column name
        label_col : str, label column name
        num_col : list, numerical feature column names
        cate_col : list, categorical feature column names (default = None)
        acutal_imp_df_path : str, path to save actual importance dataframe (true y) by csv
        null_imp_df_path : str, path to save null importance dataframe (shuffle y) by csv
        scores_df_path : str, path to save feature importance score by csv
        '''
        
        self.data = data
        self.time_col = time_col
        self.label_col = label_col
        self.num_col = num_col
        self.cate_col = cate_col
        if not self.cate_col:
            self.fea_col = self.num_col + self.cate_col
        else:
            self.fea_col = self.num_col
        self.actual_imp_df_path = actual_imp_df_path
        self.null_imp_df_path = null_imp_df_path
        self.scores_df_path = scores_df_path
        self.need_preprocess = need_preprocess
        self.shuffle = shuffle
        self.seed = seed
        self.num_boost_round = num_boost_round
        self.rounds = rounds
        
        print('feature columns:%s len:%s'%(self.fea_col, len(self.fea_col)))
        print('DATA SHAPE:', self.data.shape)
        
        if need_preprocess:
            self.preprocess_data()
            
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

            self.data = process_dataset(data)
            
        def get_feature_importances(self):
            y = self.data[self.label_col].copy()
            if self.shuffle:
                y = self.data[self.label_col].copy().sample(frac = 1.0)
            dtrain = lgb.Dataset(self.data[self.fea_col], y, free_raw_data = True)
            lgb_params = {
                'objective':'binary',
                'boosting_type':'rf',
                'subsample':0.623,
                'colsample_bytree':0.7,
                'num_leaves':127,
                'max_depth':8,
                'seed':self.seed,
                'bagging_freq':1,
                'n_jobs':-1,
                # 'data_sample_strategy':'goss',
                'verbosity':-1 
            }
    
            clf = lgb.train(params = lgb_params, train_set = dtrain, num_boost_round = self.num_boost_round, 
                            categorical_feature = self.cate_col, callbacks = [lgb.log_evaluation(period = 50)])
    
            imp_df = pd.DataFrame()
            imp_df['feature'] = list(self.fea_col)
            imp_df['importance_gain'] = clf.feature_importance(importance_type = 'gain')
            imp_df['importance_split'] = clf.feature_importance(importance_type = 'split')
            imp_df['auc_score'] = roc_auc_score(y, clf.predict(self.data[self.fea_col]))
            imp_df = imp_df.sort_values(by = ['importance_gain', 'importance_split'], ascending = [False, False])
            
            return imp_df
        
        def get_feature_importances_df(self):
            start0 = time.time()
            actual_imp_df = self.get_feature_importances()
            time_cost0 = (time.time() - start0)/60
            print('Round 0 finished')
            print('Time taken %.2f mins' %time_cost0)

            if self.actual_imp_df_path:
                actual_imp_df.to_csv(self.actual_imp_df_path)

            self.shuffle = True
            null_imp_df = pd.DataFrame()
            start = time.time()
            for i in range(self.rounds):
                imp_df = self.get_feature_importances()
                null_imp_df = pd.concat([null_imp_df, imp_df], axis = 0)
                if self.null_imp_df_path:
                    null_imp_df.to_csv(self.null_imp_df_path)
                time_cost = (time.time() - start)/60
                print('Round %s finished' %(i + 1))
                print('Total time taken %.2f mins' %time_cost)

            feature_scores = []
            for _f in actual_imp_df['feature'].unique():
                f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
                f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
                gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))
                f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
                f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
                split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))
                feature_scores.append((_f, split_score, gain_score))
                scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
                scores_df = scores_df.sort_values(by = ['gain_score', 'split_score'], ascending=[False, False]).reset_index(drop = True)
                if self.scores_df_path:
                    scores_df.to_csv(self.scores_df_path)

            return actual_imp_df, null_imp_df, scores_df