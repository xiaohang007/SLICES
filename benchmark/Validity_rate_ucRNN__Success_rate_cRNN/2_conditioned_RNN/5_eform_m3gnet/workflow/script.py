# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,csv,glob
from jarvis.io.vasp.outputs import Outcar, Vasprun
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.composition import Composition

import tensorflow as tf
from contextlib import contextmanager
from functools import wraps
import json
import signal
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())
def function_timeout(seconds: int):
    """Wrapper of Decorator to pass arguments"""
    def decorator(func):
        @contextmanager
        def time_limit(seconds_):
            def signal_handler(signum, frame):  # noqa
                raise SystemExit("Timed out!")  #TimeoutException
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds_)
            try:
                yield
            finally:
                signal.alarm(0)
        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_limit(seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator

with open('temp_splited.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            s = Structure.from_str(fmt='poscar',input_string='\n'.join(row[3].split('\\n')))
            finder = SpacegroupAnalyzer(s)
            try:
                primitive_standard_structure = finder.get_primitive_standard_structure()
            except Exception as e:
                print("Error: get_primitive_standard_structure failed!!!",e)
                primitive_standard_structure = s
            # calculate eform
            comp = primitive_standard_structure.composition
            enthalpyForm=float(row[2])*comp.num_atoms
            temp=comp.get_el_amt_dict()
            for i in range(len(temp)):
                enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
            enthalpyForm_per_atom=enthalpyForm/comp.num_atoms
            with open("result.csv", 'a') as fn:
                fn.write(row[0]+','+primitive_standard_structure.to(fmt="poscar").replace('\n','\\n')+','+row[1]+','+row[2]+','+str(enthalpyForm_per_atom)+'\n')
        except Exception as e:
            print(e)

