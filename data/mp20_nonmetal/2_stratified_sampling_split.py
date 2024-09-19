import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from slices.utils import adaptive_dynamic_binning


# 读取数据
data = pd.read_csv('mp20_eform_bandgap_nonmetal.csv')
target_column = data.columns[-1]  # 假设最后一列是目标值

# 进行自适应动态分箱
train_data, test_data, bins = adaptive_dynamic_binning(data, target_column)

# 检查分布
print("\n训练集分布:")
print(train_data[target_column].value_counts(bins=bins, normalize=True).sort_index())
print("\n测试集分布:")
print(test_data[target_column].value_counts(bins=bins, normalize=True).sort_index())

# 保存分割后的数据
train_data.to_csv('train_data_auto_binned.csv', index=False)
test_data.to_csv('test_data_auto_binned.csv', index=False)

# 打印bins信息
print(f"\n使用的bins数量: {len(bins) - 1}")
print("Bins边界:")
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    print(f"Bin {i+1}: {low:.4f} to {high:.4f}")

# 计算每个bin的数据点数量
bin_counts = pd.cut(data[target_column], bins=bins, labels=[f'Bin{i+1}' for i in range(len(bins)-1)], include_lowest=True).value_counts().sort_index()
print("\n每个bin的数据点数量:")
print(bin_counts)

# 计算最小和最大bin的数据点数量差异
min_count = bin_counts.min()
max_count = bin_counts.max()
print(f"\n最小bin的数据点数量: {min_count}")
print(f"最大bin的数据点数量: {max_count}")
print(f"最大和最小bin之间的数据点数量差异: {max_count - min_count}")
print(f"最大和最小bin之间的数据点数量比率: {max_count / min_count:.2f}")