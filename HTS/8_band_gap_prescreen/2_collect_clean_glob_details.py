#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob




pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,energy_above_hull_prescreen,band_gap_prescreen\n")

with open("results_"+pwd.split("/")[-1]+"filtered"+".csv",'w') as result:
    result.write("index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,energy_above_hull_prescreen,band_gap_prescreen\n")

result_csv=''
result_filtered_csv=''

for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            result_csv+=i
            if  0.1 < float(i.split(',')[-1]) :
                result_filtered_csv+=i
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)

with open("results_"+pwd.split("/")[-1]+"filtered"+".csv",'a') as result:
    result.write(result_filtered_csv)

#for i in glob.glob("job_*"):
    #os.system("rm -r "+i)
                
                
                
                
