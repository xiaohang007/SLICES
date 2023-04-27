# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,json,gc,math
import csv
import pickle
from pymatgen.core.composition import Composition



with open("formula_reduced_unique_all.pickle", "rb") as fp:   # Unpickling
    formula_reduced_unique_all = pickle.load(fp)
os.system("rm result.csv")  # to deal with slurm's twice execution bug

with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        comp = Composition(row[1])
        reduced_comp,factor=comp.get_reduced_composition_and_factor()
        if str(reduced_comp).replace(' ', '') not in formula_reduced_unique_all:
            with open("result.csv", 'a') as f:
                f.write(','.join(row)+'\n')





