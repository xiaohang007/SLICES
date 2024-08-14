# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math




def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system("rm result.csv")  # to deal with slurm's twice execution bug
  # This loads the default pre-trained model

with open('temp.csv', 'r') as f:
    slices_list=f.readlines()

batch_size=50
slices_split=list(split_list(slices_list,math.ceil(len(slices_list)/batch_size)))
for i in range(len(slices_split)):
    with open('temp_splited.csv', 'w') as f:
        f.writelines(slices_split[i])
    os.system("timeout "+str(batch_size* 8)+"s python -B script.py")
