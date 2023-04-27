#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob



pwd=os.getcwd()

with open("results_"+pwd.split("/")[-1]+"_filtered.csv",'w')as f:
    f.write("index,SLICES,POSCAR,formula,energy_per_atom_IAP,energy_per_atom_sym_IAP,space_group_number_IAP,dissimilarity,\
    energy_above_hull_IAP,band_gap_alignn,space_group_number_PBE,energy_above_hull_PBE,energy_per_atom_PBE,dir_gap,indir_gap\n")

result_filtered_csv=''
with open("results_"+pwd.split("/")[-1]+".csv",'r')as result:
    for i in result.readlines()[1:]:
        if len(i.split(','))==15 and float(i.split(',')[-1])==float(i.split(',')[-2]) and 0.1<= float(i.split(',')[-1]) <= 0.55:
            result_filtered_csv+=i
        

with open("results_"+pwd.split("/")[-1]+"_filtered.csv",'a')as f:
    f.write(result_filtered_csv)


                
                
                
                
