# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import sys,os,csv
import time
import argparse
import pickle,json
from monty.serialization import loadfn, dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN,BrunnerNN_reciprocal,EconNN,MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
os.system("rm result.csv")
training_fingerprint_list=loadfn("../training_fingerprint_list.json.gz")
ssf = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
    stats=('mean', 'std_dev', 'minimum', 'maximum'))
with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            temp=100
            structure = Structure.from_str(fmt='poscar',input_string='\n'.join(row[2].split('\\n')))
            vector = np.array(ssf.featurize(structure))
            
            for i in training_fingerprint_list:
                diff=np.linalg.norm(vector - i)
                if diff < temp:
                    temp=diff
            with open("result.csv", 'a') as fn:
                fn.write(','.join(row)+','+str(temp)+'\n')
        except Exception as e:
            print(e)

