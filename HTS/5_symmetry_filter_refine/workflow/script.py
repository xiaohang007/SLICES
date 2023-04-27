# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,csv,glob
from jarvis.io.vasp.outputs import Outcar, Vasprun
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from m3gnet.models import Relaxer
import tensorflow as tf
from contextlib import contextmanager
from functools import wraps
import signal
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

optimizer="BFGS"
fmax=0.2
steps=100
relaxer = Relaxer(optimizer=optimizer)
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
    
@function_timeout(seconds=100)
def m3gnet_relax(struc):
    """
    cell optimization using M3GNet IAPs (time limit is set to 200 seconds 
    to prevent buggy cell optimization that takes fovever to finish)
    """
    relax_results = relaxer.relax(struc,fmax=fmax,steps=steps)
    final_structure = relax_results['final_structure']
    final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(struc))
    return final_structure,final_energy_per_atom
with open('temp_splited.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            s = Structure.from_str(fmt='poscar',input_string='\n'.join(row[3].split('\\n')))
            finder = SpacegroupAnalyzer(s,symprec=0.5,angle_tolerance=15)
            space_group_number=finder.get_space_group_number()
            if space_group_number > 1:
                refined_structure = finder.get_refined_structure()
                refined_structure_opt, final_energy_per_atom=m3gnet_relax(refined_structure)
                finder = SpacegroupAnalyzer(refined_structure_opt)
                try:
                    primitive_standard_structure = finder.get_primitive_standard_structure()
                except Exception as e:
                    print("Error: get_primitive_standard_structure failed!!!",e)
                    primitive_standard_structure = refined_structure_opt
                with open("result.csv", 'a') as fn:
                    fn.write(row[0]+','+primitive_standard_structure.to(fmt="poscar").replace('\n','\\n')+','+row[1]+','+row[2]+','+str(final_energy_per_atom)+','+str(space_group_number)+'\n')
                
            else:
                with open("result.csv", 'a') as fn:
                    fn.write(row[0]+','+row[3]+','+row[1]+','+row[2]+','+row[2]+','+str(1)+'\n')
        except Exception as e:
            print(e)

