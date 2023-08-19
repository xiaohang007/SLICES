#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob

ehull=0.1


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("index,SCILES,POSCAR,formula,energy_per_atom,eform,eform_PBE\n")

result_csv=''

index=0
for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        lines=result.readlines()
        for j in range(len(lines)):
            result_csv+=lines[j]
            index+=1
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)

import pandas as pd

# 读取CSV文件
input_file = "results_"+pwd.split("/")[-1]+".csv"


df = pd.read_csv(input_file)




#for i in glob.glob("job_*"):
    #os.system("rm -r "+i)
                
                
                
                
