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
data=[]               
for f in glob.glob("./**/output.json", recursive=True):
    with open(f,"r") as infile:
        temp=json.load(infile)  # put each cifs into the final list
        for i in temp:
            data.append(i)
with open("../cifs_filtered.json",'w') as outfile:
    json.dump(data, outfile)     

for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                           
                
                
