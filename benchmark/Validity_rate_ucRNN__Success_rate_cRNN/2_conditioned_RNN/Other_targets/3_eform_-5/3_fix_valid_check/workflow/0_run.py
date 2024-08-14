# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
import csv
import pickle

from slices.core import SLICES

CG=SLICES()

os.system("rm result.csv")  # to deal with slurm's twice execution bug

with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            CG.from_SLICES(row[0],fix_duplicate_edge=True)
            SLICES_fixed=CG.to_SLICES()
            if CG.check_SLICES(SLICES_fixed):
                with open("result.csv", 'a') as f:
                    f.write(SLICES_fixed+'\n')
        except Exception as e:
            print(e)





