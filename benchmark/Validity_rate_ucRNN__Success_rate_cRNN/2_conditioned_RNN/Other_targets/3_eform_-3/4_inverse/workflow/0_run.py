# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math




def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system("rm result.csv")  # to deal with slurm's twice execution bug
  # This loads the default pre-trained model

with open('temp.sci', 'r') as f:
    sciles_list=f.readlines()

batch_size=30
sciles_split=list(split_list(sciles_list,math.ceil(len(sciles_list)/batch_size)))
for i in range(len(sciles_split)):
    with open('temp_splited.sci', 'w') as f:
        f.writelines(sciles_split[i])
    os.system("timeout "+str(batch_size* 16)+"s python -B script.py")
