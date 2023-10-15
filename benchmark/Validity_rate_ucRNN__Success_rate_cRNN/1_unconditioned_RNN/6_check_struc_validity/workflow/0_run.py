# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
import csv
import pickle

from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
CG=InvCryRep()

os.system("rm result.csv")  # to deal with slurm's twice execution bug

with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            with open("./temp.vasp","w") as fn:
                fn.write('\n'.join(row[3].split('\\n')))
            struc_temp=Structure.from_file("./temp.vasp")
            if CG.check_structural_validity(struc_temp):
                with open("result.csv", 'a') as f:
                    f.write(row[0]+'\n')
        except Exception as e:
            print(e)





