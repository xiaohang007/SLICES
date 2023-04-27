#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob

e_formation_limit=-0.4

pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+"filtered_"+str(e_formation_limit)+"eV.csv",'w') as result:
    result.write('SLICES,POSCAR,energy_per_atom,energy_of_formation_per_atom\n')


result_filtered_csv=''
with open("results_4_EOF.csv",'r') as result:
    for i in result.readlines()[1:]:
        if float(i.split(',')[-1]) < e_formation_limit:
            result_filtered_csv+=i
        

with open("results_"+pwd.split("/")[-1]+"filtered_"+str(e_formation_limit)+"eV.csv",'a') as result:
    result.write(result_filtered_csv)


                
                
                
                
