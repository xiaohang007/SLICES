# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob
import re
import numpy as np
import math,json


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system('rm -rf job_* structures_ori_opt ./result.csv')
#os.mkdir("structures_ori_opt")
threads=32
os.system("rm result.csv")
with open('prior_aug_50x.sci', 'r') as f:
    cifs=f.readlines()
length_list=[]
for i in cifs:
    temp=len(i.split(" "))
    with open("result.csv",'a') as fn:
        fn.write(str(temp)+'\n')
    
