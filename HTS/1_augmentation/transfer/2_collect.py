# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob,json


with open('./results.csv','w')as f:
    f.write('name,structureMatcher_bool,structureGraph_distance\n')
result_csv=''

for root,dirs,files in os.walk('./'):
    for f in files:
        if f.endswith('result.csv') :
            with open(os.path.join(root, f),'r') as result:
                result_csv += result.read()
with open("results.csv",'a') as result:
    result.write(result_csv)
result_sli=""
for root,dirs,files in os.walk('./'):
    for f in files:
        if f.endswith('result.sli') :
            with open(os.path.join(root, f),'r') as result:
                result_sli += result.read()
with open("../transfer_aug.sli",'w') as result:
    result.write(result_sli)

for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                           
                
                
