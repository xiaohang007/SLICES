# -*- coding: utf-8 -*-
import os,sys,json

from pymatgen.core.structure import Structure
import configparser
os.environ["OMP_NUM_THREADS"] = "1"

config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file
bond_scaling = config.getfloat("Settings","bond_scaling") 
delta_theta = config.getfloat("Settings","delta_theta") 
delta_x = config.getfloat("Settings","delta_x") 
lattice_shrink = config.getfloat("Settings","lattice_shrink") 
lattice_expand = config.getfloat("Settings","lattice_expand") 
angle_weight = config.getfloat("Settings","angle_weight") 
epsilon = config.getfloat("Settings","epsilon") 
repul = config.getboolean("Settings","repul") 
graph_method = config.get("Settings","graph_method")
print(delta_theta,lattice_expand)
with open('cifs_filtered.json', 'r') as f:
    cifs=json.load(f)
cifs_filtered=[]
print(len(cifs))





