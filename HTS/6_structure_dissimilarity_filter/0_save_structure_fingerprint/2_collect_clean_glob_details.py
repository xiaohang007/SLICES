#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob
from monty.serialization import loadfn, dumpfn
ehull=0.1


pwd=os.getcwd()

training_fingerprint_list=[]


for i in glob.glob("job_*/training_fingerprint_list.json.gz"):
    training_fingerprint_list+=loadfn(i)

dumpfn(training_fingerprint_list, "../training_fingerprint_list.json.gz") # or .json.gz                
                
                
