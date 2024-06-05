#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn

import os,sys,glob

pwd=os.getcwd()
os.system("rm results_"+pwd.split("/")[-1]+".csv")
result_csv=''
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("eform_target,SLICES,eform_m3gnet,poscar\n")

for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            if 1:
                result_csv+=i
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)
with open("results_"+pwd.split("/")[-1]+".csv", 'r') as f:
    cifs=f.readlines()
cifs_filtered=[]
print(len(cifs))
for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                
                
                
                
