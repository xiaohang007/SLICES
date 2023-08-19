# -*- coding: utf-8 -*-
import os,sys,json,gc,math
import csv
from pymatgen.core.composition import Composition

os.system("rm result.csv")  # to deal with slurm's twice execution bug
with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())
with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            comp = Composition(row[1])
            enthalpyForm=float(row[2])*comp.num_atoms
            temp=comp.get_el_amt_dict()
            for i in range(len(temp)):
                enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
            enthalpyForm_per_atom=enthalpyForm/comp.num_atoms
            with open("result.csv", 'a') as f:
                f.write(row[0]+','+row[3]+','+row[1]+','+row[2]+','+str(enthalpyForm_per_atom)+'\n')
        except Exception as e:
            print(e)




