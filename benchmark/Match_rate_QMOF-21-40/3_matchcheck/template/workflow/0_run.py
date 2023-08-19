# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,json,gc
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import configparser
import time

os.environ["OMP_NUM_THREADS"] = "1"


os.environ["XTB_MOD_PATH"] = "/crystal/xtb_noring_nooutput_nostdout_noCN"
config = configparser.ConfigParser()
config.read('../settings.ini') #path of your .ini file
bond_scaling = config.getfloat("Settings","bond_scaling") 
delta_theta = config.getfloat("Settings","delta_theta") 
delta_x = config.getfloat("Settings","delta_x") 
lattice_shrink = config.getfloat("Settings","lattice_shrink") 
lattice_expand = config.getfloat("Settings","lattice_expand") 
angle_weight = config.getfloat("Settings","angle_weight") 
vbond_param_ave_covered = config.getfloat("Settings","vbond_param_ave_covered") 
vbond_param_ave = config.getfloat("Settings","vbond_param_ave") 
repul = config.getboolean("Settings","repul") 
graph_method = config.get("Settings","graph_method")
print(delta_theta,lattice_expand)

  # This loads the default pre-trained model
with open('temp.json', 'r') as f:
    cifs=json.load(f)
os.system("rm result.csv")  # to deal with slurm's twice execution bug
check=False
CG=InvCryRep(graph_method=graph_method, check_results=check)
for i  in range(len(cifs)):
    p = cifs[i]["cif"] #path to CIF file
    try:
        start_time = time.time()
        ori = Structure.from_str(p,"cif")
        num_atoms=len(ori.atomic_numbers)
        CG.from_cif(p)
        #print(bond_scaling, delta_theta, delta_x,lattice_shrink,lattice_expand,angle_weight,epsilon,repul)
        structures,energy=CG.to_structures(bond_scaling, delta_theta, delta_x, \
        lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        if len(structures)==3:
            a,b,c,d,e,f=CG.match_check3(ori,structures[2],structures[1],structures[0])
            a2,b2,c2,d2,e2,f2=CG.match_check3(ori,structures[2],structures[1],structures[0],ltol=0.3, stol=0.5, angle_tol=10)
            time_used=(time.time() - start_time)
            with open("result.csv",'a') as fn:
                fn.write(cifs[i]["qmof_id"]+','+a+','+b+','+c+','+a2+','+b2+','+c2+','+d2+','+e2+','+f2+','+str(num_atoms)+','+str(time_used)+'\n')
        if len(structures)==2:
            a,b,c,d=CG.match_check(ori,structures[1],structures[0])
            a2,b2,c2,d2=CG.match_check(ori,structures[1],structures[0],ltol=0.3, stol=0.5, angle_tol=10)
            time_used=(time.time() - start_time)
            with open("result.csv",'a') as fn:
                fn.write(cifs[i]["qmof_id"]+','+'0'+','+a+','+b+','+'0'+','+a2+','+b2+','+'1'+','+c2+','+d2+','+str(num_atoms)+','+str(time_used)+'\n')
    except Exception as e1:
        del CG
        CG=InvCryRep(graph_method=graph_method, check_results=check)
        with open("result.csv",'a') as fn:
            fn.write(cifs[i]["qmof_id"]+','+str(e1).split('\n')[0]+'\n')

