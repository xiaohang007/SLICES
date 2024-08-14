import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split(data, target_column, test_size=0.1, random_state=42):
    # 尝试将目标列转换为数值型，并捕获无法转换的值
    non_numeric = data[pd.to_numeric(data[target_column], errors='coerce').isna()]
    
    if not non_numeric.empty:
        print("以下行无法转换为数值类型:")
        print(non_numeric)
        print("\n无法转换的唯一值:")
        print(non_numeric[target_column].unique())
    
    # 将目标列转换为数值型，设置errors='coerce'将非数值转换为NaN
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    
    # 删除目标列中的NaN值
    data_cleaned = data.dropna(subset=[target_column])
    
    print(f"\n原始数据行数: {len(data)}")
    print(f"清理后数据行数: {len(data_cleaned)}")
    
    # 将目标值分成区间
    data_cleaned['bin'] = pd.cut(data_cleaned[target_column], 
                         bins=[-np.inf, 0, 0.5, 1, 2, np.inf], 
                         labels=['zero', 'low', 'medium', 'high', 'very_high'])
    
    train_data = pd.DataFrame(columns=data_cleaned.columns)
    test_data = pd.DataFrame(columns=data_cleaned.columns)
    
    # 对每个区间进行分层抽样
    for bin_label in data_cleaned['bin'].unique():
        bin_data = data_cleaned[data_cleaned['bin'] == bin_label]
        if len(bin_data) > 1:
            bin_train, bin_test = train_test_split(bin_data, test_size=test_size, random_state=random_state)
        else:
            bin_train, bin_test = bin_data, pd.DataFrame()
        
        train_data = pd.concat([train_data, bin_train])
        test_data = pd.concat([test_data, bin_test])
    
    # 删除临时的'bin'列
    train_data = train_data.drop('bin', axis=1)
    test_data = test_data.drop('bin', axis=1)
    
    # 打乱数据
    train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_data, test_data

# 读取数据
data = pd.read_csv('mp20_eform_bandgap_nonmetal.csv')
target_column = data.columns[-1]  # 假设最后一列是目标值

# 进行分层抽样
train_data, test_data = stratified_split(data, target_column)

# 检查分布
print("\n训练集分布:")
print(train_data[target_column].value_counts(bins=5, normalize=True))
print("\n测试集分布:")
print(test_data[target_column].value_counts(bins=5, normalize=True))

# 保存分割后的数据
train_data.to_csv('train_data_reduce_zero.csv', index=False)
test_data.to_csv('test_data_reduce_zero.csv', index=False)
