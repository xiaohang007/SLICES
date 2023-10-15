#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob


ehull=0.1


pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write("index,SLICES,POSCAR,formula,energy_per_atom_IAP,energy_per_atom_sym_IAP,space_group_number_IAP,dissimilarity,\
    energy_above_hull_IAP,band_gap_alignn,space_group_number_PBE,energy_above_hull_PBE,energy_per_atom_PBE,dir_gap,indir_gap\n")

result_csv=''

os.system("rm -r candidates")
os.mkdir("candidates")
for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            result_csv+=i

for i in glob.glob("job_*/candidates/*.png"):
    os.system("cp "+i+" ./candidates")

with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)



#for i in glob.glob("job_*"):
    #os.system("rm -r "+i)
                
                
