import pandas as pd
import numpy as np

### 对于下列所有函数，都可以使用toad原生库进行计算
### toad计算分箱时，通常是每一个唯一值分一箱，为了减少计算量可以预分箱
### 特征分箱并计算woe、iv、ks等基础指标
def feature_stats_by_bin(df, feature, label, bin_nums = 10, precision = 3, if_format = True):
    data = df.copy()
    data[feature] = pd.to_numeric(data[feature], errors= 'coerce')
    data[label] = data[label].astype('int')
    data = data[data[label].isin([0, 1, 2])]
    data[label] = data[label].apply(lambda x:1 if x == 1 else 0)
    
    _, bins = pd.cut(data[feature], bins = bin_nums, retbins = True)
    bins[0], bins[-1] = -np.inf, np.inf
    bins = np.unique(bins)
    data['bin'] = pd.cut(data[feature], bins = bins, include_lowest = True)
    
    ### 统计正样本及负样本数量
    cut_group_all = data.groupby('bin', dropna = False).agg({label:'count'})
    cut_y = data.groupby('bin', dropna = False).agg({label:'sum'})
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
    
    ### 对于dataframe中的字段标准化处理
    if if_format:
        tp['bad_count'] = tp['bad_count'].astype(int)
        tp['good_count'] = tp['good_count'].astype(int)
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
    print(feature_stats_by_bin(df, 'feature', 'label'))
    
    print(-9999 == -9999)