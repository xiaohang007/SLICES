#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn

import os,sys,glob

pwd=os.getcwd()
with open("results_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write('name,opt2_match,opt_match,std2_match,std_match,opt2_match2,opt_match2,std2_match2,std_match2,opt2_topo2,opt_topo2,std2_topo2,std_topo2,natoms,time\n')
result_csv=''


for i in glob.glob("job_*/result.csv"):
    with open(i,'r') as result:
        for i in result.readlines():
            if len(i.split(','))==15:
                result_csv+=i
        
with open("results_"+pwd.split("/")[-1]+".csv",'a') as result:
    result.write(result_csv)

#for i in glob.glob("job_*"):
#    os.system("rm -r "+i)
                
                
                
                

