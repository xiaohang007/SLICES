#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob

band_gap_target=0.1

pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+"filtered_"+str(band_gap_target)+"eV.csv",'w') as result:
    result.write('SLICES,POSCAR,energy_per_atom,energy_of_formation_per_atom\n')


result_filtered_csv=''
with open("results_8_band_gap_prescreen.csv",'r') as result:
    for i in result.readlines()[1:]:
        if  band_gap_target<float(i.split(',')[-1]):
            result_filtered_csv+=i
        

with open("results_"+pwd.split("/")[-1]+"filtered_"+str(band_gap_target)+"eV.csv",'a') as result:
    result.write(result_filtered_csv)


                
                
                
                
