#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com

import os,sys,glob

pwd=os.getcwd()

result_csv=''


for i in glob.glob("job_*/100x.sci"):
    with open(i,'r') as result:
        for i in result.readlines():
            result_csv+=i
        
with open("sampled_"+pwd.split("/")[-1]+".sli",'w') as result:
    result.write(result_csv)

for i in glob.glob("job_*"):
    os.system("rm -r "+i)
                
                
                
                
