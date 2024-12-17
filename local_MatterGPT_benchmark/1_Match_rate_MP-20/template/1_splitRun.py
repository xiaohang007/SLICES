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
threads=30


with open('../../../data/mp20/cifs_filtered.json', 'r') as f:
    cifs=json.load(f)
cifs_split=list(split_list(cifs,threads))

for i in range(len(cifs_split)):
    os.mkdir('job_'+str(i))
    os.system('cp -r ./workflow/. job_'+str(i))
    with open('temp.json', 'w') as f:
        json.dump(cifs_split[i], f)
    os.system('mv temp.json job_'+str(i))
    os.chdir('job_'+str(i))
    os.system('qsub 0_run.pbs')
    os.chdir('..')
    #os.system('cp -rf GWP_Serial/.reholu_template Z'+i)


