# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,json,gc,math
import csv
import pickle

from invcryrep.invcryrep import InvCryRep

CG=InvCryRep()

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





