#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write('SCILES,POSCAR,energy_per_atom\n')

os.system("rm results_"+pwd.split("/")[-1]+".sli")

result_csv=''
result_filtered_csv=''
slices=[]

for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            slices.append(i.strip())



with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    for i in slices:
        result.write(i+'\n')
with open("results_"+pwd.split("/")[-1]+".csv", 'r') as f:
    cifs=f.readlines()
cifs_filtered=[]
print(len(cifs)-1)

for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                
                
                
                
