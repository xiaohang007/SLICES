#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob

ehull=0.2

pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+"filtered_"+str(ehull)+"eV.csv",'w') as result:
    result.write("index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,dissimilarity\n")


result_filtered_csv=''
with open("results_2_EOFfiltered_-1.5eV_0.85ev_0.5radius.csv",'r') as result:
    for i in result.readlines()[1:]:
        if  float(i.split(',')[-1]) < ehull:
            result_filtered_csv+=i
        

with open("results_"+pwd.split("/")[-1]+"filtered_"+str(ehull)+"eV.csv",'a') as result:
    result.write(result_filtered_csv)


                
                
                
                
