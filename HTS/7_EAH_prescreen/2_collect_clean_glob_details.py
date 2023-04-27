#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob

ehull=0.1


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,dissimilarity,energy_above_hull_prescreen\n")

result_csv=''


for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            result_csv+=i
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)



#for i in glob.glob("job_*"):
    #os.system("rm -r "+i)
                
                
                
                
