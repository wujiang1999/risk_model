import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

### 对于列进行归一化操作
def normalize(df, columns = None, method = 'minmax'):
    '''
    df: DataFrame
    columns:list, default None, columns to be normalized
    method: str, default 'minmax', 'minmax' or 'standard', how to normalize
    '''
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("mode method be 'minmax' or 'standard'")
        
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    return df_normalized

### 对于缺失值处理 将缺失值填充为0并且将原序列double 添加数据标识对应位置取值是否为空
def add_nan_indicator(array, type = 'float32'):
    '''
    array: numpy array
    '''
    array = np.array(array)
    
    if not np.isnan(array).any():
        return array.astype(type)
    
    indicator = (~np.isnan(array)).astype(int)
    new_array = np.hstack((array, indicator))
    new_array = np.nan_to_num(new_array, nan = 0.0)
    return new_array.astype(type)

### 创建训练用的DataLoader
def create_dataloader(X, y, batch_size = 32, shuffle = True, need_nan_process = True):
    '''
    X: numpy array
    y: numpy array
    '''
    if need_nan_process:
        X = add_nan_indicator(X)
    X_tensor = torch.tensor(X, dtype = torch.float32)
    y_tensor = torch.tensor(y, dtype = torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    # print(len(dataset))
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

### 定义一个早停函数
class EarlyStopping:
    def __init__(self, metric = 'val_auc', round = 10, min_delta = 1e-3, 
                 verbose = False, restore_best_weights = True, mode = 'max'):
        self.metric = metric
        self.round = round
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.count = 0
        self.early_stop = False
        self.best_weights = None
        
        if self.mode == 'min':
            self.is_improvement = lambda current, best:current < best - self.min_delta
        elif self.mode == 'max':
            self.is_improvement = lambda current, best:current > best + self.min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'")
        
    ### __call__允许对象的实例可以像函数一样被调用
    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                print(f'Initial {self.metric} is set as {self.best_score:.4f}')
            if self.restore_best_weights:
                self.best_weights = self.get_weights(model.state_dict())
        elif self.is_improvement(current_score, self.best_score):
            if self.verbose:
                improvement = 'raise' if self.mode == 'max' else 'fall'
                print(f'{self.metric} {improvement} from {self.best_score:.4f} to {current_score:.4f}, early stopping count reset')
            if self.restore_best_weights:
                self.best_weights = self.get_weights(model.state_dict())
            self.best_score = current_score
            self.count = 0
        else:
            self.count += 1
            if self.verbose:
                print(f'Early stopping count {self.count}/{self.round}')
            if self.count >= self.round:
                if self.verbose:
                    print(f'{self.metric} early stopping triggered after {self.count} rounds')
                self.early_stop = True
            
    def get_weights(self, state_dict):
        return {k:v.clone().detach() for k, v in state_dict.items()}
    
    def load_best_weights(self, model):
        if self.best_weights:
            model.load_state_dict(self.best_weights)

### 定义一个简单的模型结构
class DNN(nn.Module):
    def __init__(self, input_dim, num_classes = 2, dropout_p = 0.1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_p)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x) 
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    arr_with_nan = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    arr_without_nan = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(np.isnan(arr_with_nan).any(), np.isnan(arr_without_nan).any())
    # print(add_nan_indicator(arr_with_nan), add_nan_indicator(arr_without_nan))
    
    def generate_dataset(num_samples = 1000, num_features = 10, nan_ratio = 0.05, imbalance_ratio = 0.7):
        # 生成随机特征数据，标准正态分布
        X = np.random.randn(num_samples, num_features)

        # 计算要引入的NaN数量
        total_elements = num_samples * num_features
        num_nans = int(total_elements * nan_ratio)

        # 随机选择位置引入NaN
        nan_indices = np.unravel_index(
            np.random.choice(total_elements, num_nans, replace=False),
            (num_samples, num_features)
        )
        X[nan_indices] = np.nan

        # 生成不平衡的随机标签
        y = np.random.choice([0, 1], size = num_samples, p = [1 - imbalance_ratio, imbalance_ratio])

        return X, y

    X_train, y_train = generate_dataset(imbalance_ratio = 0.7)
    X_valid, y_valid = generate_dataset(imbalance_ratio = 0.65)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    train_dataloader, valid_dataloader = create_dataloader(X_train, y_train), create_dataloader(X_valid, y_valid)
    
    input_dim = len(X_train[0]) * 2
    # print(input_dim)
    num_classes = 2
    model = DNN(input_dim = input_dim, num_classes = num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    early_stopping = EarlyStopping(metric = 'val_auc', round = 20, min_delta = 1e-4, verbose = True, mode = 'max')
    for epoch in range(1000):
    # 训练阶段
        model.train()
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(features)  # 输出 logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for features, labels in valid_dataloader:
                outputs = model(features)  # 输出 logits
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                # 计算概率
                probs = torch.softmax(outputs, dim = 1)[:, 1]  # 类别1的概率
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(valid_dataloader.dataset)
        
        # 计算 AUC
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0  # 当所有标签相同或其他情况导致 AUC 计算失败时
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # 检查早停条件
        early_stopping(val_auc, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 恢复最佳权重
    if early_stopping.restore_best_weights:
        early_stopping.load_best_weights(model)
        print("Loaded best weights")
    

    
    
