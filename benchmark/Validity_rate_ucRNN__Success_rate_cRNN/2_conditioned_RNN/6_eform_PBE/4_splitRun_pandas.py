# -*- coding: utf-8 -*-
import os,sys,glob
import re
import numpy as np
import math,json
import pandas as pd

with open('./results_6_ok.csv', 'r') as f:
    lines=f.readlines()
pick=[]
for i in lines:
    pick.append(i.split(",")[0])

os.system('rm -rf job_* structures_ori_opt ./result.csv')
#os.mkdir("structures_ori_opt")




def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system('rm -rf job_* structures_ori_opt ./result.csv temp.csv')
#os.mkdir("structures_ori_opt")
threads=30


with open('./results_5_eform_m3gnet.csv', 'r') as f:
    cifs=f.readlines()
cifs_pick=[]
for i in cifs:
    if i.split(",")[0] not in pick:
        cifs_pick.append(i)

cifs_split=list(split_list(cifs_pick,threads))

for i in range(len(cifs_split)):
    os.mkdir('job_'+str(i))
    os.system('cp -r ./workflow/. job_'+str(i))
    with open('temp.csv', 'w') as f:
        f.writelines(cifs_split[i])
    os.system('cp temp.csv job_'+str(i))

    os.chdir('job_'+str(i))
    if len(sys.argv)==2:
        if sys.argv[1]=="test":
            os.system('yhbatch 0_test.pbs')
    else:
        os.system('yhbatch 0_run.sh')
    os.chdir('..')
os.system('rm temp.csv')
