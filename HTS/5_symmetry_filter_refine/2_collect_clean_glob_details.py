#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob

ehull=0.1


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number\n")

result_csv=''

index=0
for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        lines=result.readlines()
        for j in range(len(lines)):
            result_csv+=str(index)+','+lines[j]
            index+=1
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)

import pandas as pd

# 读取CSV文件
input_file = "results_"+pwd.split("/")[-1]+".csv"
output_file = "results_"+pwd.split("/")[-1]+"_filtered.csv"

df = pd.read_csv(input_file)

# 对第二列进行分组，然后筛选每组中第四列最大（不能等于1）的条目，最后筛选每组中第三列最小的条目
result = df.loc[df['space_group_number'] != 1].groupby(['formula','space_group_number'], group_keys=False).apply(lambda x: x[x.energy_per_atom_sym==x.energy_per_atom_sym.min()])
result.to_csv(output_file, index=False)


for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                
                
                
                
