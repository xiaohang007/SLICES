import pandas as pd
import os,sys,glob

pwd=os.getcwd()
# 读取CSV文件
results_1 = pd.read_csv("results_"+pwd.split("/")[-1]+".csv")
downstream = pd.read_csv('../../1_downstream_dataset/train_downstream.sli',header=None)
os.system("rm energy_formation_m3gnet_lists.csv")
# 提取必要的列
header_values = results_1.iloc[:, 0].tolist()
data_values = results_1.iloc[:, 2].tolist()
downstream_values = downstream.iloc[:, 1].tolist()

# 创建一个字典，用于存储数据
data_dict = {}

# 将results_1的第一列作为header，并将第三列的数据存入字典
for header, value in zip(header_values, data_values):
    if header in data_dict:
        data_dict[header].append(value)
    else:
        data_dict[header] = [value]


sorted_keys = sorted(data_dict.keys(), reverse=True)

# 根据排序后的键的顺序创建DataFrame
energy_formation_m3gnet_lists = pd.DataFrame({k: pd.Series(data_dict[k], index=range(len(data_dict[k]))) for k in sorted_keys})



# 添加downstream的第二列作为新的列，列名为'mp-20'
energy_formation_m3gnet_lists = pd.concat([energy_formation_m3gnet_lists, pd.Series(downstream_values, name='training_dataset')], axis=1)

# 保存结果到新的CSV文件
energy_formation_m3gnet_lists.to_csv('energy_formation_m3gnet_lists.csv', index=False)

print("程序执行完毕，结果已保存到energy_formation_m3gnet_lists_final.csv")
