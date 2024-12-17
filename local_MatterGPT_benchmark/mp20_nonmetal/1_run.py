# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
from utils import *
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from slices.utils import adaptive_dynamic_binning

output=[]
data_path_predix="../mp20/"
data=pd.read_csv(data_path_predix+"test.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
data=pd.read_csv(data_path_predix+"val.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
data=pd.read_csv(data_path_predix+"train.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
with open('cifs.json', 'w') as f:
    json.dump(output, f)

splitRun_local(filename='./cifs.json',threads=8,skip_header=False)
show_progress_local()
collect_json(output="cifs_filtered.json", \
    glob_target="./**/output.json",cleanup=False)
collect_csv(output="mp20_eform_bandgap_nonmetal.csv", \
    glob_target="./**/result.csv",cleanup=True,header="SLICES,eform,bandgap\n")
os.system("rm cifs.json")



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
train_data.to_csv('train_data_reduce_zero.csv', index=False)
test_data.to_csv('test_data_reduce_zero.csv', index=False)

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