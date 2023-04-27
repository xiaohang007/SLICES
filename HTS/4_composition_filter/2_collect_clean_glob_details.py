#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write('SLICES,POSCAR,energy_per_atom\n')



result_csv=''
result_filtered_csv=''

for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            result_csv+=i
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)



for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                
                
                
                
