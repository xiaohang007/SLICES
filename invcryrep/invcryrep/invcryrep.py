# -*- coding: UTF-8 -*-
import os,subprocess,random,warnings
warnings.filterwarnings("ignore")
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN,BrunnerNN_reciprocal,EconNN,MinimumDistanceNN
from pymatgen.core.periodic_table import ElementBase
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
import networkx as nx
from networkx.algorithms import tree
import numpy as np
from invcryrep.tobascco_net import Net, SystreDB
import math
import tempfile
import json
from scipy.optimize import fsolve,fmin_l_bfgs_b
from collections import defaultdict, deque
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from m3gnet.models import Relaxer
import logging
import tensorflow as tf
import signal,gc
from contextlib import contextmanager
from functools import wraps
logging.captureWarnings(False)
tf.get_logger().setLevel(logging.ERROR)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# define a decorator that sets a time limit for a function call
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

# 3*3*3 supercell: 27 cells in -1, 0, 1 offsets in the x, y, z dimensions
offset = np.array([
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
])
# lj_params of the repulsive potentials for uncovered pairs
lj_params_list = [
[1, 2.20943,0.552357],
[2, 1.99561, 0.498903],
[3, 9.1228, 2.2807],
[4, 6.8421, 1.71053],
[5, 6.05811, 1.51453],
[6, 5.41666, 1.35417],
[7, 5.0603, 1.26508],
[8, 4.70395, 1.17599],
[9, 4.0625, 1.01562],
[10, 4.13377, 1.03344],
[11, 11.8311, 2.95778],
[12, 10.0493, 2.51233],
[13, 8.6239, 2.15597],
[14, 7.91118, 1.9778],
[15, 7.62609, 1.90652],
[16, 7.48355, 1.87089],
[17, 7.26973, 1.81743],
[18, 7.55482, 1.88871],
[19, 14.4682, 3.61705],
[20, 12.5439, 3.13596],
[21, 12.1162, 3.02906],
[22, 11.4035, 2.85088],
[23, 10.9046, 2.72615],
[24, 9.90679, 2.4767],
[25, 9.90679, 2.4767],
[26, 9.40789, 2.35197],
[27, 8.98026, 2.24506],
[28, 8.83772, 2.20943],
[29, 9.40789, 2.35197],
[30, 8.69517, 2.17379],
[31, 8.69517, 2.17379],
[32, 8.55263, 2.13816],
[33, 8.48136, 2.12034],
[34, 8.55263, 2.13816],
[35, 8.55263, 2.13816],
[36, 8.26754, 2.06689],
[37, 15.6798, 3.91995],
[38, 13.898, 3.47451],
[39, 13.5417, 3.38542],
[40, 12.4726, 3.11815],
[41, 11.6886, 2.92215],
[42, 10.9759, 2.74397],
[43, 10.477, 2.61924],
[44, 10.4057, 2.60142],
[45, 10.1206, 2.53015],
[46, 9.90679, 2.4767],
[47, 10.3344, 2.58361],
[48, 10.2632, 2.56579],
[49, 10.1206, 2.53015],
[50, 9.90679, 2.4767],
[51, 9.90679, 2.4767],
[52, 9.83552, 2.45888],
[53, 9.90679, 2.4767],
[54, 9.97807, 2.49452],
[55, 17.3903, 4.34759],
[56, 15.3235, 3.83086],
[57, 14.7533, 3.68832],
[58, 14.5395, 3.63487],
[59, 14.4682, 3.61705],
[60, 14.3257, 3.58141],
[61, 14.1831, 3.54578],
[62, 14.1118, 3.52796],
[63, 14.1118, 3.52796],
[64, 13.9693, 3.49232],
[65, 13.8267, 3.45669],
[66, 13.6842, 3.42105],
[67, 13.6842, 3.42105],
[68, 13.4704, 3.3676],
[69, 13.5417, 3.38542],
[70, 13.3278, 3.33196],
[71, 13.3278, 3.33196],
[72, 12.4726, 3.11815],
[73, 12.1162, 3.02906],
[74, 11.546, 2.88651],
[75, 10.7621, 2.69051],
[76, 10.2632, 2.56579],
[77, 10.0493, 2.51233],
[78, 9.69298, 2.42324],
[79, 9.69298, 2.42324],
[80, 9.40789, 2.35197],
[81, 10.3344, 2.58361],
[82, 10.4057, 2.60142],
[83, 10.5482, 2.63706],
[84, 9.97807, 2.49452],
[85, 10.6908, 2.6727],
[86, 10.6908, 2.6727],
[87, 18.5307, 4.63267],
[88, 15.7511, 3.93777],
[89, 15.3235, 3.83086],
[90, 14.682, 3.6705],
[91, 14.2544, 3.56359],
[92, 13.9693, 3.49232],
[93, 13.5417, 3.38542],
[94, 13.3278, 3.33196],
[95, 12.8289, 3.20724],
[96, 12.045, 3.01124],
[97, 11.9737, 2.99342],
[98, 11.9737, 2.99342],
[99, 11.7599, 2.93997],
[100, 11.9024, 2.9756],
[101, 12.33 ,3.08251],
[102, 12.5439, 3.13596],
[103, 11.4748, 2.86869],
[104, 11.1897, 2.79742],
[105, 10.6195, 2.65488],
[106, 10.1919, 2.54797],
[107, 10.0493, 2.51233],
[108, 9.55043, 2.38761],
[109, 9.19407, 2.29852],
[110, 9.1228, 2.2807],
[111, 8.6239, 2.15597],
[112, 8.69517, 2.17379],
[113, 9.69298, 2.42324],
[114, 10.1919, 2.54797],
[115, 11.546, 2.88651],
[116, 12.4726, 3.11815],
[117, 11.7599, 2.93997],
[118, 11.1897, 2.79742],
]

# periodic table data
periodic_data = pd.read_csv(
    StringIO(
        r'''
atomic_number,symbol,name,atomic_mass,electron_configuration,phase,category,period,group,mass_number,radioactive,valence_electrons
1,H,Hydrogen,1.00794,1s1,gas,nonmetal,1,IA,1,False,1
2,He,Helium,4.002602,1s2,gas,noble gas,1,VIIIA,4,False,0
3,Li,Lithium,6.941,[He] 2s1,solid,alkali metal,2,IA,7,False,1
4,Be,Beryllium,9.012182,[He] 2s2,solid,alkaline earth metal,2,IIA,9,False,2
5,B,Boron,10.811,[He] 2s2 2p1,solid,metalloid,2,IIIA,11,False,3
6,C,Carbon,12.0107,[He] 2s2 2p2,solid,nonmetal,2,IVA,12,False,4
7,N,Nitrogen,14.0067,[He] 2s2 2p3,gas,nonmetal,2,VA,14,False,5
8,O,Oxygen,15.9994,[He] 2s2 2p4,gas,nonmetal,2,VIA,16,False,6
9,F,Fluorine,18.9984032,[He] 2s2 2p5,gas,halogen,2,VIIA,19,False,7
10,Ne,Neon,20.1797,[He] 2s2 2p6,gas,noble gas,2,VIIIA,20,False,0
11,Na,Sodium,22.98976928,[Ne] 3s1,solid,alkali metal,3,IA,23,False,1
12,Mg,Magne-sium,24.3050,[Ne] 3s2,solid,alkaline earth metal,3,IIA,24,False,2
13,Al,Aluminum,26.9815386,[Ne] 3s2 3p1,solid,post-transition metal,3,IIIA,27,False,3
14,Si,Silicon,28.0855,[Ne] 3s2 3p2,solid,metalloid,3,IVA,28,False,4
15,P,Phosphor-us,30.973762,[Ne] 3s2 3p3,solid,nonmetal,3,VA,31,False,5
16,S,Sulfur,32.065,[Ne] 3s2 3p4,solid,nonmetal,3,VIA,32,False,6
17,Cl,Chlorine,35.453,[Ne] 3s2 3p5,gas,halogen,3,VIIA,35,False,7
18,Ar,Argon,39.948,[Ne] 3s2 3p6,gas,noble gas,3,VIIIA,40,False,0
19,K,Potassium,39.0983,[Ar] 4s1,solid,alkali metal,4,IA,39,False,1
20,Ca,Calcium,40.078,[Ar] 4s2,solid,alkaline earth metal,4,IIA,40,False,2
21,Sc,Scandium,44.955912,[Ar] 3d1 4s2,solid,transition metal,4,IIIB,45,False,3
22,Ti,Titanium,47.867,[Ar] 3d2 4s2,solid,transition metal,4,IVB,48,False,4
23,V,Vanadium,50.9415,[Ar] 3d3 4s2,solid,transition metal,4,VB,51,False,5
24,Cr,Chromium,51.9961,[Ar] 3d5 4s1,solid,transition metal,4,VIB,52,False,6
25,Mn,Manga-nese,54.938045,[Ar] 3d5 4s2,solid,transition metal,4,VIIB,55,False,7
26,Fe,Iron,55.845,[Ar] 3d6 4s2,solid,transition metal,4,VIIIB,56,False,8
27,Co,Cobalt,58.933195,[Ar] 3d7 4s2,solid,transition metal,4,VIIIB,59,False,9
28,Ni,Nickel,58.6934,[Ar] 3d8 4s2,solid,transition metal,4,VIIIB,58,False,10
29,Cu,Copper,63.546,[Ar] 3d10 4s1,solid,transition metal,4,IB,63,False,11
30,Zn,Zinc,65.38,[Ar] 3d10 4s2,solid,transition metal,4,IIB,64,False,12
31,Ga,Gallium,69.723,[Ar] 3d10 4s2 4p1,solid,post-transition metal,4,IIIA,69,False,3
32,Ge,German-ium,72.64,[Ar] 3d10 4s2 4p2,solid,metalloid,4,IVA,74,False,4
33,As,Arsenic,74.92160,[Ar] 3d10 4s2 4p3,solid,metalloid,4,VA,75,False,5
34,Se,Selenium,78.96,[Ar] 3d10 4s2 4p4,solid,nonmetal,4,VIA,80,False,6
35,Br,Bromine,79.904,[Ar] 3d10 4s2 4p5,liquid,halogen,4,VIIA,79,False,7
36,Kr,Krypton,83.798,[Ar] 3d10 4s2 4p6,gas,noble gas,4,VIIIA,84,False,0
37,Rb,Rubidium,85.4678,[Kr] 5s1,solid,alkali metal,5,IA,85,False,1
38,Sr,Strontium,87.62,[Kr] 5s2,solid,alkaline earth metal,5,IIA,88,False,2
39,Y,Yttrium,88.90585,[Kr] 4d1 5s2,solid,transition metal,5,IIIB,89,False,3
40,Zr,Zirconium,91.224,[Kr] 4d2 5s2,solid,transition metal,5,IVB,90,False,4
41,Nb,Niobium,92.90638,[Kr] 4d4 5s1,solid,transition metal,5,VB,93,False,5
42,Mo,Molybde-num,95.96,[Kr] 4d5 5s1,solid,transition metal,5,VIB,98,False,6
43,Tc,Technetium,98,[Kr] 4d5 5s2,solid,transition metal,5,VIIB,98,True,7
44,Ru,Ruthenium,101.07,[Kr] 4d7 5s1,solid,transition metal,5,VIIIB,102,False,8
45,Rh,Rhodium,102.90550,[Kr] 4d8 5s1,solid,transition metal,5,VIIIB,103,False,9
46,Pd,Palladium,106.42,[Kr] 4d10,solid,transition metal,5,VIIIB,106,False,12
47,Ag,Silver,107.8682,[Kr] 4d10 5s1,solid,transition metal,5,IB,107,False,11
48,Cd,Cadmium,112.411,[Kr] 4d10 5s2,solid,transition metal,5,IIB,114,False,12
49,In,Indium,114.818,[Kr] 4d10 5s2 5p1,solid,post-transition metal,5,IIIA,115,False,3
50,Sn,Tin,118.710,[Kr] 4d10 5s2 5p2,solid,post-transition metal,5,IVA,120,False,4
51,Sb,Antimony,121.760,[Kr] 4d10 5s2 5p3,solid,metalloid,5,VA,121,False,5
52,Te,Tellurium,127.60,[Kr] 4d10 5s2 5p4,solid,metalloid,5,VIA,130,False,6
53,I,Iodine,126.90447,[Kr] 4d10 5s2 5p5,solid,halogen,5,VIIA,127,False,7
54,Xe,Xenon,131.293,[Kr] 4d10 5s2 5p6,gas,noble gas,5,VIIIA,132,False,0
55,Cs,Cesium,132.9054519,[Xe] 6s1,solid,alkali metal,6,IA,133,False,1
56,Ba,Barium,137.327,[Xe] 6s2,solid,alkaline earth metal,6,IIA,138,False,2
57,La,Lanthanum,138.90547,[Xe] 5d1 6s2,solid,lanthanide,6,,139,False,2
58,Ce,Cerium,140.116,[Xe] 4f1 5d1 6s2,solid,lanthanide,6,,140,False,2
59,Pr,Praseo-dymium,140.90765,[Xe] 4f3 6s2,solid,lanthanide,6,,141,False,2
60,Nd,Neo-dymium,144.242,[Xe] 4f4 6s2,solid,lanthanide,6,,142,False,2
61,Pm,Prome-thium,145,[Xe] 4f5 6s2,solid,lanthanide,6,,145,True,2
62,Sm,Samarium,150.36,[Xe] 4f6 6s2,solid,lanthanide,6,,152,False,2
63,Eu,Europium,151.964,[Xe] 4f7 6s2,solid,lanthanide,6,,153,False,2
64,Gd,Gadolinium,157.25,[Xe] 4f7 5d1 6s2,solid,lanthanide,6,,158,False,2
65,Tb,Terbium,158.92535,[Xe] 4f9 6s2,solid,lanthanide,6,,159,False,2
66,Dy,Dyspro-sium,162.500,[Xe] 4f10 6s2,solid,lanthanide,6,,164,False,2
67,Ho,Holmium,164.93032,[Xe] 4f11 6s2,solid,lanthanide,6,,165,False,2
68,Er,Erbium,167.259,[Xe] 4f12 6s2,solid,lanthanide,6,,166,False,2
69,Tm,Thulium,168.93421,[Xe] 4f13 6s2,solid,lanthanide,6,,169,False,2
70,Yb,Ytterbium,173.054,[Xe] 4f14 6s2,solid,lanthanide,6,,174,False,2
71,Lu,Lutetium,174.9668,[Xe] 4f14 5d1 6s2,solid,transition metal,6,IIIB,175,False,3
72,Hf,Hafnium,178.49,[Xe] 4f14 5d2 6s2,solid,transition metal,6,IVB,180,False,4
73,Ta,Tantalum,180.94788,[Xe] 4f14 5d3 6s2,solid,transition metal,6,VB,181,False,5
74,W,Tungsten,183.84,[Xe] 4f14 5d4 6s2,solid,transition metal,6,VIB,184,False,6
75,Re,Rhenium,186.207,[Xe] 4f14 5d5 6s2,solid,transition metal,6,VIIB,187,False,7
76,Os,Osmium,190.23,[Xe] 4f14 5d6 6s2,solid,transition metal,6,VIIIB,192,False,8
77,Ir,Iridium,192.217,[Xe] 4f14 5d7 6s2,solid,transition metal,6,VIIIB,193,False,9
78,Pt,Platinum,195.084,[Xe] 4f14 5d9 6s1,solid,transition metal,6,VIIIB,195,False,10
79,Au,Gold,196.966569,[Xe] 4f14 5d10 6s1,solid,transition metal,6,IB,197,False,11
80,Hg,Mercury,200.59,[Xe] 4f14 5d10 6s2,liquid,transition metal,6,IIB,202,False,12
81,Tl,Thallium,204.3833,[Xe] 4f14 5d10 6s2 6p1,solid,post-transition metal,6,IIIA,205,False,3
82,Pb,Lead,207.2,[Xe] 4f14 5d10 6s2 6p2,solid,post-transition metal,6,IVA,208,False,4
83,Bi,Bismuth,208.98040,[Xe] 4f14 5d10 6s2 6p3,solid,post-transition metal,6,VA,209,False,5
84,Po,Polonium,,[Xe] 4f14 5d10 6s2 6p4,solid,metalloid,6,VIA,209,True,6
85,At,Astatine,,[Xe] 4f14 5d10 6s2 6p5,solid,halogen,6,VIIA,210,True,7
86,Rn,Radon,,[Xe] 4f14 5d10 6s2 6p6,gas,noble gas,6,VIIIA,222,True,0
87,Fr,Francium,,[Rn] 7s1,solid,alkali metal,7,IA,223,True,1
88,Ra,Radium,,[Rn] 7s2,solid,alkaline earth metal,7,IIA,226,True,2
89,Ac,Actinium,,[Rn] 6d1 7s2,solid,actinide,7,,227,True,2
90,Th,Thorium,232.03806,[Rn] 6d2 7s2,solid,actinide,7,,232,True,2
91,Pa,Protact-inium,231.03588,[Rn] 5f2 6d1 7s2,solid,actinide,7,,231,True,2
92,U,Uranium,238.02891,[Rn] 5f3 6d1 7s2,solid,actinide,7,,238,True,2
93,Np,Neptunium,,[Rn] 5f4 6d1 7s2,solid,actinide,7,,237,True,2
94,Pu,Plutonium,,[Rn] 5f6 7s2,solid,actinide,7,,244,True,2
95,Am,Americium,,[Rn] 5f7 7s2,solid,actinide,7,,243,True,2
96,Cm,Curium,,[Rn] 5f7 6d1 7s2,solid,actinide,7,,247,True,2
97,Bk,Berkelium,,[Rn] 5f9 7s2,solid,actinide,7,,247,True,2
98,Cf,Californ-ium,,[Rn] 5f10 7s2,solid,actinide,7,,251,True,2
99,Es,Einstein-ium,,[Rn] 5f11 7s2,solid,actinide,7,,252,True,2
100,Fm,Fermium,,[Rn] 5f12 7s2,,actinide,7,,257,True,2
101,Md,Mendelev-ium,,[Rn] 5f13 7s2,,actinide,7,,258,True,2
102,No,Nobelium,,[Rn] 5f14 7s2,,actinide,7,,259,True,2
103,Lr,Lawrenc-ium,,[Rn] 5f14 7s2 7p1,,transition metal,7,IIIB,262,True,3
104,Rf,Ruther-fordium,,[Rn] 5f14 6d2 7s2,,transition metal,7,IVB,261,True,4
105,Db,Dubnium,,[Rn] 5f14 6d3 7s2,,transition metal,7,VB,262,True,5
106,Sg,Seaborg-ium,,[Rn] 5f14 6d4 7s2,,transition metal,7,VIB,266,True,6
107,Bh,Bohrium,,[Rn] 5f14 6d5 7s2,,transition metal,7,VIIB,264,True,7
108,Hs,Hassium,,[Rn] 5f14 6d6 7s2,,transition metal,7,VIIIB,269,True,8
109,Mt,Meitnerium,,[Rn] 5f14 6d7 7s2,,transition metal,7,VIIIB,268,True,9
110,Ds,Darmstadt-ium,,[Rn] 5f14 6d9 7s1,,transition metal,7,VIIIB,,True,10
111,Rg,Roentgen-ium,,[Rn] 5f14 6d10 7s1,,transition metal,7,IB,,True,11
112,Cn,Copernic-ium,,[Rn] 5f14 6d10 7s2,,transition metal,7,IIB,,True,12
113,Nh,Nihonium,,[Rn] 5f14 6d10 7s2 7p1,,unknown,7,IIIA,,True,3
114,Fl,Flerovium,,[Rn] 5f14 6d10 7s2 7p2,,unknown,7,IVA,,True,4
115,Mc,Moscov-ium,,[Rn] 5f14 6d10 7s2 7p3,,unknown,7,VA,,True,5
116,Lv,Livermor-ium,,[Rn] 5f14 6d10 7s2 7p4,,unknown,7,VIA,,True,6
117,Ts,Tennessine,,[Rn] 5f14 6d10 7s2 7p5,,unknown,7,VIIA,,True,7
118,Og,Oganesson,,[Rn] 5f14 6d10 7s2 7p6,,unknown,7,VIIIA,,True,0
'''
    ),
    dtype=str,
)

class InvCryRep:
    def __init__(self, atom_types=None, edge_indices=None, to_jimages=None, graph_method='econnn',check_results=False, optimizer="BFGS",fmax=0.2,steps=100):
        """
        atom_types: np.array
        edge_indices: np.array
        to_jimages: np.array
        graph_method: string
        check_results: bool
        atom_symbols: list
        SLICES: string
        unstable_graph: bool
        relaxer: m3gnet class
        fmax: float
        steps: int
        """    
        tf.keras.backend.clear_session()
        gc.collect()
        self.atom_types = atom_types
        self.edge_indices = edge_indices
        self.to_jimages = to_jimages
        self.graph_method = graph_method
        self.check_results = check_results  # for debug
        self.atom_symbols = None
        self.SLICES = None
        self.unstable_graph = False  # unstable graph flag
        self.relaxer = Relaxer(optimizer=optimizer)
        self.fmax=fmax
        self.steps=steps

    def check_element(self):
        # make sure no atoms with atomic numbers higher than 86 (gfnff)
        if self.atom_types.max() < 87:
            return True
        else:
            return False

    def file2structure_graph(self,filename):
        """
        convert file to pymatgen structure_graph object
        """
        structure = Structure.from_file(filename)
        if self.graph_method == 'brunnernn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal())
        elif self.graph_method == 'econnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN())
        elif self.graph_method == 'mininn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN())
        elif self.graph_method == 'crystalnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN())
        else:
            print("ERROR - graph_method not implemented") 
        return structure_graph,structure

    def cif2structure_graph(self,string):
        """
        convert a cif string to structure_graph
        """
        structure = Structure.from_str(string,'cif')
        if self.graph_method == 'brunnernn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal())
        elif self.graph_method == 'econnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN())
        elif self.graph_method == 'mininn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN())
        elif self.graph_method == 'crystalnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN())
        else:
            print("ERROR - graph_method not implemented") 
        return structure_graph,structure

    def structure2structure_graph(self,structure):
        """
        convert pymatgen structure to structure_graph
        """
        if self.graph_method == 'brunnernn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal())
        elif self.graph_method == 'econnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN())
        elif self.graph_method == 'mininn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN())
        elif self.graph_method == 'crystalnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN())
        else:
            print("ERROR - graph_method not implemented") 
        return structure_graph

    def from_SLICES(self,SLICES):
        """
        extract edge_indices, to_jimages and atom_types from decoding a SLICES string
        """
        tokens=SLICES.split(" ")
        for i in range(len(tokens)):
            if tokens[i].isnumeric():
                num_atoms=i
                break
        self.atom_symbols=tokens[:num_atoms]
        num_edges=int((len(tokens)-len(self.atom_symbols))/5)
        edge_indices=np.zeros([num_edges,2],dtype=int)
        to_jimages=np.zeros([num_edges,3],dtype=int)
        for i in range(num_edges):
            edge=tokens[num_atoms+i*5:num_atoms+(i+1)*5]
            edge_indices[i,0]=int(edge[0])
            edge_indices[i,1]=int(edge[1])
            for j in range(3):
                if edge[j+2]=='-':
                    to_jimages[i,j]=-1
                if edge[j+2]=='o':
                    to_jimages[i,j]=0
                if edge[j+2]=='+':
                    to_jimages[i,j]=1
        self.edge_indices=edge_indices
        self.to_jimages=to_jimages
        self.atom_types=np.array([int(periodic_data.loc[periodic_data["symbol"]==i].values[0][0]) for i in self.atom_symbols])        


    def structure2SLICES(self,structure,strategy=3):
        """
        extract edge_indices, to_jimages and atom_types from a pymatgen structure object
         then encoding them into a SLICES string
        """        
        structure_graph=self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        def get_slices1(atom_symbols,edge_indices,to_jimages):
            atom_symbols_mod = [ (i+'_')[:2] for i in atom_symbols]
            SLICES=""
            for i in range(len(edge_indices)):
                SLICES+=atom_symbols_mod[edge_indices[i][0]]+('0'+str(edge_indices[i][0]))[-2:]+atom_symbols_mod[edge_indices[i][1]]+('0'+str(edge_indices[i][1]))[-2:]
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='-'
                    if j==0:
                        SLICES+='o'
                    if j==1:
                        SLICES+='+'
            return SLICES
        def get_slices2(atom_symbols,edge_indices,to_jimages):
            atom_symbols_mod = [ (i+'_')[:2] for i in atom_symbols]
            SLICES=""
            for i in atom_symbols_mod:
                SLICES+=i
            for i in range(len(edge_indices)):
                SLICES+=('0'+str(edge_indices[i][0]))[-2:]+('0'+str(edge_indices[i][1]))[-2:]
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='-'
                    if j==0:
                        SLICES+='o'
                    if j==1:
                        SLICES+='+'
            return SLICES
        def get_slices3(atom_symbols,edge_indices,to_jimages):
            SLICES=''
            for i in atom_symbols:
                SLICES+=i+' '
            for i in range(len(edge_indices)):
                SLICES+=str(edge_indices[i][0])+' '+str(edge_indices[i][1])+' '
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='- '
                    if j==0:
                        SLICES+='o '
                    if j==1:
                        SLICES+='+ '
            return SLICES
        if strategy==1:
            return get_slices1(atom_symbols,edge_indices,to_jimages)
        if strategy==2:
            return get_slices2(atom_symbols,edge_indices,to_jimages)
        if strategy==3:
            return get_slices3(atom_symbols,edge_indices,to_jimages)

    def structure2SLICESAug(self,structure,strategy=3,num=40):
        """
        (1) extract edge_indices, to_jimages and atom_types from a pymatgen structure object
        (2) encoding edge_indices, to_jimages and atom_types into multiple equalivent SLICES strings 
        with a data augmentation scheme
        """
        structure_graph=self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        # shuffle to get permu
        permu=[]
        for i in range(num):
            permu.append(tuple(random.sample(atom_symbols, k=len(atom_symbols))))
        permu_unique=list(set(permu))
        #print(permu_unique)
        # For duplicates, we take the smallest index that has not been taken.
        def get_index_list_allow_duplicates(ori,mod):
            indexes = defaultdict(deque)
            for i, x in enumerate(mod):
                indexes[x].append(i)
            ids = [indexes[x].popleft() for x in ori]
            return ids
        index_mapping=[]
        for i in permu_unique:
            if i != tuple(atom_symbols):
                index_mapping.append(get_index_list_allow_duplicates(atom_symbols,i))
        def get_slices1(atom_symbols,edge_indices,to_jimages):
            atom_symbols_mod = [ (i+'_')[:2] for i in atom_symbols]
            SLICES=""
            for i in range(len(edge_indices)):
                SLICES+=atom_symbols_mod[edge_indices[i][0]]+('0'+str(edge_indices[i][0]))[-2:]+atom_symbols_mod[edge_indices[i][1]]+('0'+str(edge_indices[i][1]))[-2:]
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='-'
                    if j==0:
                        SLICES+='o'
                    if j==1:
                        SLICES+='+'
            return SLICES
        def get_slices2(atom_symbols,edge_indices,to_jimages):
            atom_symbols_mod = [ (i+'_')[:2] for i in atom_symbols]
            SLICES=""
            for i in atom_symbols_mod:
                SLICES+=i
            for i in range(len(edge_indices)):
                SLICES+=('0'+str(edge_indices[i][0]))[-2:]+('0'+str(edge_indices[i][1]))[-2:]
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='-'
                    if j==0:
                        SLICES+='o'
                    if j==1:
                        SLICES+='+'
            return SLICES
        def get_slices3(atom_symbols,edge_indices,to_jimages):
            SLICES=''
            for i in atom_symbols:
                SLICES+=i+' '
            for i in range(len(edge_indices)):
                SLICES+=str(edge_indices[i][0])+' '+str(edge_indices[i][1])+' '
                for j in to_jimages[i]:
                    if j==-1:
                        SLICES+='- '
                    if j==0:
                        SLICES+='o '
                    if j==1:
                        SLICES+='+ '
            return SLICES
        SLICES_list=[]
        if strategy==1:
            SLICES_list.append(get_slices1(atom_symbols,edge_indices,to_jimages))
        if strategy==2:
            SLICES_list.append(get_slices2(atom_symbols,edge_indices,to_jimages))
        if strategy==3:
            SLICES_list.append(get_slices3(atom_symbols,edge_indices,to_jimages))
        def shuffle_dual_list(a,b):
            c = list(zip(a, b))
            random.shuffle(c)
            a2, b2 = zip(*c)
            return a2,b2
        for i in range(len(index_mapping)):
            atom_symbols_new=permu_unique[i]
            edge_indices_new=edge_indices.copy()
            for j in range(len(edge_indices)):
                edge_indices_new[j][0]=index_mapping[i][edge_indices[j][0]]
                edge_indices_new[j][1]=index_mapping[i][edge_indices[j][1]]
            edge_indices_new_shuffled,to_jimages_shuffled=shuffle_dual_list(edge_indices_new,to_jimages)
            if strategy==1:
                SLICES_list.append(get_slices1(atom_symbols_new,edge_indices_new_shuffled,to_jimages_shuffled))
            if strategy==2:
                SLICES_list.append(get_slices2(atom_symbols_new,edge_indices_new_shuffled,to_jimages_shuffled))
            if strategy==3:
                SLICES_list.append(get_slices3(atom_symbols_new,edge_indices_new_shuffled,to_jimages_shuffled))
        return SLICES_list

    def get_dim(self,structure):
        """
        get the dimension of a structure
        """
        if self.graph_method == 'brunnernn':
            bonded_structure = BrunnerNN_reciprocal().get_bonded_structure(structure)
        elif self.graph_method == 'econnn':
            bonded_structure = EconNN().get_bonded_structure(structure)
        elif self.graph_method == 'mininn':
            bonded_structure = MinimumDistanceNN().get_bonded_structure(structure)
        elif self.graph_method == 'crystalnn':
            bonded_structure = CrystalNN().get_bonded_structure(structure)
        else:
            print("ERROR - graph_method not implemented") 
        dim=get_dimensionality_larsen(bonded_structure)
        return dim

    def from_cif(self, string):
        """
        extract edge_indices, to_jimages and atom_types from a cif string
        """        
        structure_graph,structure = self.cif2structure_graph(string)
        if self.check_results:
            structure_graph.draw_graph_to_file('graph.png',hide_image_edges = False)
        self.atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        self.edge_indices = np.array(edge_indices)
        self.to_jimages = np.array(to_jimages)

    def from_file(self, filename):
        """
        extract edge_indices, to_jimages and atom_types from a file
        """            
        structure_graph,structure = self.file2structure_graph(filename)
        if self.check_results:
            structure_graph.draw_graph_to_file('graph.png',hide_image_edges = False)
        self.atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        self.edge_indices = np.array(edge_indices)
        self.to_jimages = np.array(to_jimages)

    def from_structure(self, structure):
        """
        extract edge_indices, to_jimages and atom_types from a pymatgen structure object
        """    
        structure_graph = self.structure2structure_graph(structure)
        if self.check_results:
            structure_graph.draw_graph_to_file('graph.png',hide_image_edges = False)
        self.atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        self.edge_indices = np.array(edge_indices)
        self.to_jimages = np.array(to_jimages)

    def structure2crystal_graph_rep(self, structure):
        """
        convert a pymatgen structure object into the crystal graph representation (atom_types,edge_indices,to_jimages)
        """
        structure_graph = self.structure2structure_graph(structure)
        if self.check_results:
            structure_graph.draw_graph_to_file('graph.png',hide_image_edges = False)
        atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=structure_graph.graph.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3:
            print("ERROR - could not deal with graph with rank H1(X,Z) < 3") # cannot generate 3D embedding
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        return atom_types,np.array(edge_indices),np.array(to_jimages)

    def get_nbf_blist(self):
        """
        get nbf(neighbor list with atom types for xtb_mod): str
        get blist(node indexes of the central unit cell edges in the 3*3*3 supercell): np.array
        """
        if self.atom_types is not None and self.edge_indices is not None and self.to_jimages is not None:
            n_atom=len(self.atom_types)
            voltage=np.concatenate((self.edge_indices, self.to_jimages), axis=1) # voltage is actually [i j volatge]
            #print(voltage)
            voltage_super=[] # [i j volatge] array for the supercell
            for i in range(len(offset)):
                for j in range(len(voltage)):
                    temp=[]
                    temp.append(voltage[j,0]+i*n_atom)   
                    voltage_sum=voltage[j,2:]+offset[i,:]
                    voltage_sum_new=voltage_sum.copy()
                    for k in range(len(voltage_sum)):
                        if voltage_sum[k] == 2:
                            voltage_sum_new[k] = 1
                        elif voltage_sum[k] == -2:
                            voltage_sum_new[k] = -1
                        else:
                            voltage_sum_new[k] = 0
                    target_block = voltage_sum.copy()
                    for k in range(len(voltage_sum)):
                        if voltage_sum[k] == 2:
                            target_block[k] = voltage_sum[k]-3
                        elif voltage_sum[k] == -2:
                            target_block[k] = voltage_sum[k]+3
                    target_block=np.array(target_block)
                    for k in range(len(offset)):
                        if np.array_equal(target_block,offset[k]):
                            target_index=k
                    temp.append(voltage[j,1]+target_index*n_atom)   # calculate the index of the target atom
                    temp.append(voltage_sum_new[0])
                    temp.append(voltage_sum_new[1])
                    temp.append(voltage_sum_new[2])
                    voltage_super.append(temp)
            voltage_super=np.array(voltage_super,dtype=int)
            # get rid of boundary case
            voltage_super_cut=[]
            for i in range(len(voltage_super)):
                if voltage_super[i,2]==0 and voltage_super[i,3]==0 and voltage_super[i,4]==0:
                    temp=[]
                    temp.append(voltage_super[i,0])
                    temp.append(voltage_super[i,1])
                    voltage_super_cut.append(temp)
            voltage_super_cut=np.array(voltage_super_cut,dtype=int)
            #print((voltage_super_cut))
            # reindex voltage_super_cut due to the removal of atoms with no neighbor
            no_neighbor_index=[]
            for i in range(n_atom*27):
                row,column=np.where(voltage_super_cut==i)
                total=len(row)
                if total==0:   # no neighbor fix (used to lead to Failed to generate charges)
                    no_neighbor_index.append(i)
            atom_symbol_list_super=[]
            atom_symbols=[str(ElementBase.from_Z(i)) for i in self.atom_types]
            for i in range(len(offset)):
                for j in range(n_atom):
                    atom_symbol_list_super.append(atom_symbols[j])
            if 1:  # wheather delete atoms with no neighbor or not
                for i in range(len(voltage_super_cut)):
                    voltage_super_cut[i,0]=voltage_super_cut[i,0]-len(np.where(no_neighbor_index < voltage_super_cut[i,0])[0]) # offset index
                    voltage_super_cut[i,1]=voltage_super_cut[i,1]-len(np.where(no_neighbor_index < voltage_super_cut[i,1])[0])
                #print(voltage_super_cut)
                for i in range(len(voltage_super)):  # modify voltage_super's index 
                    voltage_super[i,0]=voltage_super[i,0]-len(np.where(no_neighbor_index < voltage_super[i,0])[0]) # offset index
                    voltage_super[i,1]=voltage_super[i,1]-len(np.where(no_neighbor_index < voltage_super[i,1])[0])
                # get atomic symbol list of the supercell
                no_neighbor_index.reverse() # .reverse() only update the original list (no return value)
                for i in no_neighbor_index:
                    del atom_symbol_list_super[i]
            else:
                no_neighbor_index=[]
            # get nbf
            neighbor_list=np.zeros((20,n_atom*27-len(no_neighbor_index)))
            for i in range(n_atom*27-len(no_neighbor_index)):
                row,column=np.where(voltage_super_cut==i)
                total=len(row)
                if total>19:   # deal with cases with more than 19 neighbors 
                    row=row[:19]  
                    column=column[:19]
                for j in range(len(row)):
                    neighbor_list[j,i]=voltage_super_cut[row[j],1-column[j]] + 1   # nbf indexing starts with 1 instead of 0
                neighbor_list[19,i]=total

            nbf=str(n_atom*27-len(no_neighbor_index))+'\n'+' '.join(atom_symbol_list_super)+'\n'
            for i in range(20):
                nbf=nbf+' '.join([str(int(x)) for x in neighbor_list[i,:].tolist()])+'\n'
            # get blist
            blist=voltage_super[len(voltage)*13:len(voltage)*14,:2]+1  # assume that atoms with no neighbors are not in the center cell
        else:
            print("ERROR - crystalgraph is not defined") # cannot generate 3D embedding
        if self.check_results:
            with open('testBonds_cut.top','w') as f:
                f.write(nbf)
            bdict={'blist':blist.tolist()}
            with open('blist.json', 'w') as  f:
                json.dump(bdict,f)
        return nbf, blist

    def get_inner_p_target(self, bond_scaling=1.05):
        """
        get inner_p_target(inner_p matrix obtained by gfnff): np.array
        get colattice_inds(keep track of all the valid colattice dot indices): list
        get colattice_weights(colattice weights for bond or angle): list
        """
        nbf, blist = self.get_nbf_blist()
        temp_dir = tempfile.TemporaryDirectory(dir="/dev/shm")
        try:
            with open(temp_dir.name+'/testBonds_cut.top','w') as f:
                f.write(nbf)
            subprocess.call(os.environ["XTB_MOD_PATH"]+' --gfnff testBonds_cut.top --wrtopo blist,vbond,alist,vangl', \
            cwd=temp_dir.name, shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            if self.check_results:
                os.system("cp "+temp_dir.name+'/testBonds_cut.top '+os.getcwd())
                os.system("cp "+temp_dir.name+'/gfnff_lists.json '+os.getcwd())
            with open(temp_dir.name+'/gfnff_lists.json', 'r') as fgfn:
                txt=fgfn.read()
                txt=txt.replace("********","       0") # deal with the xtb output bug
                data = json.loads(txt)  # read blist,vbond,alist,vangl
                del(txt)
            temp_dir.cleanup()
            blist_original=blist
            blist_flip=np.flip(blist_original,1) # reverse the order
            bond_weight=[]
            blist_super=np.array(data['blist'],dtype=int)
            inner_p_target=np.zeros((len(blist_original),len(blist_original)))
            # bond
            bond_repul_scale=[]  # track repul scale of all bonds
            for i in range(len(blist_original)):
                #matching
                temp1=np.where((blist_super ==blist_original[i] ).all(axis=1))
                temp2=np.where((blist_super ==blist_flip[i] ).all(axis=1))
                if len(temp1[0]):
                    index=temp1[0][0]
                elif len(temp2[0]):
                    index=temp2[0][0]
                else:
                    print('Cannot find bond!!!') 
                if np.isnan(data['vbond'][index][2]):
                    bond_weight.append(1)
                    data['vbond'][index][2]=1
                else:
                    bond_weight.append(abs(data['vbond'][index][2])) # convert back to angstrom
                inner_p_target[i,i]=round((data['vbond'][index][3]*0.529177*1.05*bond_scaling)**2,5)  # *1.04, a empirical parameter
            blist_unique=np.unique(blist_original)
            alist_original=np.array(data['alist'],dtype=int)
            #print(list(alist_original))
            colattice_inds=[[],[]]
            colattice_weights=[]
            for i in range(len(alist_original)):
                ab=alist_original[i][[0,1]] # first bond
                ac=alist_original[i][[0,2]] # second bond
                #print(ab,ac)
                temp1=np.where((blist_original ==ab ).all(axis=1))
                temp2=np.where((blist_flip ==ab ).all(axis=1))
                temp3=np.where((blist_original ==ac ).all(axis=1))
                temp4=np.where((blist_flip ==ac ).all(axis=1))   
                if (len(temp1[0])+len(temp2[0])) and (len(temp3[0])+len(temp4[0])):
                    sign=1
                    if len(temp1[0]):
                        index_x=temp1[0][0]
                    elif len(temp2[0]):
                        index_x=temp2[0][0]
                        sign=sign*(-1)
                    else:
                        print('Cannot find bond1!!!') 
                    if len(temp3[0]):
                        index_y=temp3[0][0]
                    elif len(temp4[0]):
                        index_y=temp4[0][0]
                        sign=sign*(-1)
                    else:
                        print('Cannot find bond2!!!')  
                    #print(inner_p_target[index_x,index_y],index_x,index_y)
                    inner_p_target[index_x,index_y]=sign*math.sqrt(inner_p_target[index_x,index_x])*math.sqrt(inner_p_target[index_y,index_y])*math.cos(data['vangl'][i][0])
                    colattice_inds[0].append(int(index_x))
                    colattice_inds[1].append(int(index_y))
                    colattice_weights.append(abs(data['vangl'][i][1])) # vangl params could be negative numbers
            # colattice_inds weight
            temp_max=0  # angle weight is larger than 0
            for i in range(len(colattice_weights)):
                if colattice_weights[i] > temp_max:
                    temp_max=colattice_weights[i]
            for i in range(len(colattice_weights)):
                colattice_weights[i]=round(colattice_weights[i]/temp_max,2)
            temp_max=0 
            for i in range(len(bond_weight)):
                if bond_weight[i] > temp_max:
                    temp_max=bond_weight[i]
            for i in range(len(bond_weight)):
                colattice_inds[0].append(i)
                colattice_inds[1].append(i)
                colattice_weights.append(round(bond_weight[i]/temp_max,2))  # angleweight first, bondweight second
            if self.check_results:
                inner_p_target_dict={'inner_p_target':inner_p_target.tolist(),'colattice_inds':colattice_inds,'colattice_weights':colattice_weights}
                with open('inner_p_target.json', 'w') as  f:
                    json.dump(inner_p_target_dict,f)
            return inner_p_target, colattice_inds, colattice_weights
        except:
            temp_dir.cleanup()


    def convert_graph(self):
        """
        convert edge_indices, to_jimages into networkx format
        """
        edges=list(np.concatenate((self.edge_indices, self.to_jimages), axis=1))
        x_dat,net_voltage = [],[]
        for id, (v1, v2, e1, e2, e3) in enumerate(edges):
            ename = "e%i" % (id + 1)
            net_voltage.append((e1, e2, e3))
            x_dat.append((str(v1+1), str(v2+1), dict(label=ename)))  # networkx compliant
        net_voltage=np.array(net_voltage)
        return x_dat, net_voltage

    @staticmethod 
    def get_uncovered_pair(graph):  # assuming that all atoms has been included in graph 
        """
        get atom pairs not covered by edges of the structure graph 
        """
        num_nodes=len(graph.nodes)
        unique_covered_pair=[]
        covered_pair=[]
        for i in graph.edges:
            covered_pair.append(i[0]+','+i[1])  #int(i[0])-1,int(i[1])-1
        unique_covered_pair=list(dict.fromkeys(covered_pair))
        unique_covered_pair_int=[]
        for i in unique_covered_pair:
            unique_covered_pair_int.append([int(i.split(',')[0])-1,int(i.split(',')[1])-1])
        uncovered_pair=[]
        index_i,index_j=np.triu_indices(num_nodes)
        triu_pair=[]
        for i in range(len(index_i)):
            if index_i[i]!=index_j[i]:
                triu_pair.append([index_i[i],index_j[i]])
        for i in triu_pair:
            if i  not in unique_covered_pair_int:
                uncovered_pair.append(i)
        return uncovered_pair

    def get_uncovered_pair_lj(self,uncovered_pair):
        """
        get the lj parameters for atom pairs not covered by edges of the structure graph 
        """
        uncovered_pair_lj=[]
        # read lj params
        lj_param={}
        for i in lj_params_list:   
            lj_param[i[0]]=[i[1],i[2]]
        for i in uncovered_pair:
            if i[0]==i[1]:
                uncovered_pair_lj.append(lj_param[self.atom_types[i[0]]])
            else:
                uncovered_pair_lj.append([lj_param[self.atom_types[i[0]]][0]/2+lj_param[self.atom_types[i[0]]][0]/2,\
                lj_param[self.atom_types[i[0]]][1]/2+lj_param[self.atom_types[i[0]]][1]/2])
        return uncovered_pair_lj

    def get_covered_pair_lj(self):
        """
        get atom pairs covered by edges of the structure graph 
        """
        covered_pair_lj=[]
        # read lj params
        lj_param={}
        for i in lj_params_list:   
            lj_param[i[0]]=[i[1],i[2]]
        for i in self.edge_indices:
            if i[0]==i[1]:
                covered_pair_lj.append(lj_param[self.atom_types[i[0]]])
            else:
                covered_pair_lj.append([lj_param[self.atom_types[i[0]]][0]/2+lj_param[self.atom_types[i[0]]][0]/2,\
                lj_param[self.atom_types[i[0]]][1]/2+lj_param[self.atom_types[i[0]]][1]/2])
        return covered_pair_lj

    def get_rescaled_lattice_vectors(self,inner_p_target,inner_p_std,lattice_vectors_std,arc_coord_std):
        """
        get rescaled_lattice_vectors based on the GFNFF bond lengths
         calculated using the topological neighbor list as input
        """
        scaleList=[]
        inner_p_std_diag=np.diag(inner_p_std)
        nonzero_edge_index=np.where(inner_p_std_diag>0.0001)[0].tolist()
        if len(nonzero_edge_index) < len(inner_p_std_diag):
            self.unstable_graph = True  
        else:
            self.unstable_graph = False  # to refresh this setting in case of Exception
        scale_sum_temp=0
        for i in range(len(self.edge_indices)):
            if i in nonzero_edge_index:
                scaleList.append(inner_p_target[i,i]**(0.5)/inner_p_std[i,i]**(0.5))
                scale_sum_temp+=inner_p_target[i,i]**(0.5)/inner_p_std[i,i]**(0.5)
            else:
                scaleList.append(0) # placeholder for unstable graph
        scale_ave_temp=scale_sum_temp/len(nonzero_edge_index) # assume that at least one ratio is ok
        for i in range(len(self.edge_indices)):
            if i not in nonzero_edge_index:
                scaleList[i]=scale_ave_temp # replace problematic scales with the average of valid values
        #print("scaleList:",scaleList) 
        weight_sum=[0,0,0]
        scale_sum=[0,0,0]
        for i in range(len(arc_coord_std)):
            weight_sum+=abs(arc_coord_std[i])
            scale_sum+=abs(arc_coord_std[i])*scaleList[i]
        scale_vector=[0,0,0]
        for i in range(len(scale_sum)):
            scale_vector[i]=scale_sum[i]/weight_sum[i]
        #print("scale_vector",scale_vector)
        lattice_vectors_scaled = np.dot(lattice_vectors_std,np.diag(scale_vector))
        #print(arc_coord_std)
        #print("cartisian length:",np.sqrt(np.sum(np.square(np.dot(arc_coord_std,lattice_vectors_scaled)),axis=1))) # should be like this
        #print(lattice_vectors_scaled)
        return lattice_vectors_scaled

    # calcualte the fractional coordinates of vertices 
    @staticmethod 
    def get_coordinates(arc_coord,num_nodes,shortest_path_spanning_graph,spanning):
        """
        get fractional coordinates of atoms from fractional coordinates of arcs vectors
        """
        coordinates=np.zeros((num_nodes,3))  #v0 
        if num_nodes>1:  # !deal with single node case
            for i in range(num_nodes)[1:]:  # get vi
                for h in range(len(shortest_path_spanning_graph[str(i+1)][1:])):    # every edge in shortest path  # str(i) convert '0' to 0
                    for j in spanning:            
                        if set(shortest_path_spanning_graph[str(i+1)][h:h+2])==set(j[:2]):  # compare with every edge in spanning tree  
                            if int(shortest_path_spanning_graph[str(i+1)][h])== int(j[0]):                           
                                coordinates[i,:]=coordinates[i,:]+arc_coord[int(j[2][1:])-1]
                            else:
                                coordinates[i,:]=coordinates[i,:]-arc_coord[int(j[2][1:])-1]
        else:
            coordinates[0,:]=[0,0,0]
        return coordinates

    @staticmethod
    def convert_params(x, ndim, angle_inds, cocycle_size, lattice_type, metric_tensor_std):
        """
        extract metric_tensor and cocycle_rep from x vector
        """
        if lattice_type==1: 
            cell_lengths = x[:1]
            cocycle = x[1:]
            mt = np.empty((ndim, ndim))
            for i in range(ndim):
                mt[i, i] = x[0]
        elif lattice_type>20:
            cell_lengths = x[:2]
            cocycle = x[2:]
            mt = np.empty((ndim, ndim))
            if lattice_type==21:
                mt[0, 0] = x[0]
                mt[1, 1] = x[1]
                mt[2, 2] = x[1]
            if lattice_type==22:
                mt[0, 0] = x[0]
                mt[1, 1] = x[1]
                mt[2, 2] = x[0]      
            if lattice_type==23:
                mt[0, 0] = x[0]
                mt[1, 1] = x[0]
                mt[2, 2] = x[1]            
        else:
            cell_lengths = x[:3]
            cocycle = x[3:]
            mt = np.empty((ndim, ndim))   
            for i in range(ndim):
                mt[i, i] = x[i]
        g = np.triu_indices(ndim, 1)
        for (i, j) in zip(*g):
            mt[i, j] = metric_tensor_std[i, j]/np.sqrt(metric_tensor_std[i, i])/np.sqrt(metric_tensor_std[j, j])*np.sqrt(mt[i, i])*np.sqrt(mt[j, j])
            mt[j, i] = mt[i, j]                 
        if cocycle_size == 0:
            cocycle_rep = None
        else:
            cocycle_rep = np.reshape(cocycle, (cocycle_size, ndim))
        return mt, cocycle_rep
    
    @staticmethod
    def initialize_x_bounds(ndim,cocycle_rep,metric_tensor_std,lattice_type,delta_theta,delta_x,lattice_expand,lattice_shrink): # 
        """
        initialize x vectors and bounds based on metric_tensor_std, lattice_type and other settings
        """
        if lattice_type==1: 
            # equal length 
            mtsize=1
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            xinc = 0
            # non triclinic cases, no need to change lattice angles
            x[xinc] = metric_tensor_std[0,0]
            ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
            lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
            xinc += 1
        elif lattice_type>20:
            mtsize=2
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            if lattice_type==21 or lattice_type==22:
                xinc = 0
                x[xinc] = metric_tensor_std[0,0]
                ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
                xinc += 1
                x[xinc] = metric_tensor_std[1,1]
                ub[xinc] = metric_tensor_std[1,1]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[1,1]*(lattice_shrink)**2  # min_cell
                xinc += 1
            else:
                xinc = 0
                x[xinc] = metric_tensor_std[0,0]
                ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
                xinc += 1
                x[xinc] = metric_tensor_std[2,2]
                ub[xinc] = metric_tensor_std[2,2]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[2,2]*(lattice_shrink)**2  # min_cell
                xinc += 1                
        else:
            # triclinic
            mtsize=3
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            xinc = 0
            for i in range(3):
                x[xinc] = metric_tensor_std[i,i]
                ub[xinc] = metric_tensor_std[i,i]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[i,i]*(lattice_shrink)**2  # min_cell
                xinc += 1
        if cocycle_rep is not None:
            x[xinc:] = cocycle_rep.flatten()
            ub[xinc:] = delta_x
            lb[xinc:] = -1*delta_x
        bounds=[]
        for i in range(len(x)):
            bounds.append((lb[i],ub[i]))        
        return x, bounds

    @staticmethod
    def all_distances(coords1, coords2):
        """
        Returns the distances between two lists of coordinates
        Args:
            coords1: First set of Cartesian coordinates.
            coords2: Second set of Cartesian coordinates.
        Returns:
            2d array of Cartesian distances. E.g the distance between
            coords1[i] and coords2[j] is distances[i,j]
        """
        c1 = np.array(coords1)
        c2 = np.array(coords2)
        z = (c1[:, None, :] - c2[None, :, :]) ** 2
        return np.sum(z, axis=-1) ** 0.5


    def func(self,x,ndim,order,mat_target,colattice_inds,colattice_weights, \
        cycle_rep,cycle_cocycle_I,num_nodes,shortest_path,spanning,uncovered_pair, \
        uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
        lattice_vectors_scaled,structure_species,angle_weight,repul,lattice_type,metric_tensor_std):
        """
        objective function: sum squared differences between the inner products of the GFN-FF predicted 
        geometry and the associated inner products (gjk) of the edges in the non-barycentric embedded net
        """
        square_diff=0
        # convert x to inner_p
        angle_inds = int(math.factorial(ndim) / math.factorial(2) / math.factorial(ndim - 2))
        metric_tensor, cocycle_rep = self.convert_params(x, ndim, angle_inds, int(order - 1),lattice_type,metric_tensor_std)
        if cocycle_rep is not None: 
            periodic_rep = np.concatenate((cycle_rep, cocycle_rep))
        else:
            periodic_rep=cycle_rep
        lattice_arcs = np.dot(cycle_cocycle_I, periodic_rep)
        inner_p = np.dot(np.dot(lattice_arcs, metric_tensor), lattice_arcs.T)

        for k in range(len(colattice_inds[0])):
            i=colattice_inds[0][k]
            j=colattice_inds[1][k]
            if i==j:
                square_diff += (inner_p[i][j]-mat_target[i][j])**2 + 4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12  # inner_p单位是 Angstrom的平方
            else:
                square_diff += angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2  # divide angle weight by 3
        # repulsive part of LJ potential to prevent atom collision
        if repul:
            # calculate distance matrix
            coordinates_temp=self.get_coordinates(lattice_arcs,num_nodes,shortest_path,spanning)
            coordinates_temp_cart=np.dot(coordinates_temp,lattice_vectors_scaled)  # using the orginal scaled lattice vectors to simplify calculation
            distance_matrix=self.all_distances(coordinates_temp_cart,coordinates_temp_cart)
            for i in range(len(uncovered_pair)):  #  cutoff        # epsilon       # sigma\
                r_temp=distance_matrix[uncovered_pair[i][0],uncovered_pair[i][1]]
                if r_temp<uncovered_pair_lj[i][0]:
                    square_diff+=4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12 
        return square_diff

    def func_check(self,x,ndim,order,mat_target,colattice_inds,colattice_weights,cycle_rep,cycle_cocycle_I,num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave,lattice_vectors_scaled,structure_species,angle_weight,repul,lattice_type,metric_tensor_std):
        """
        print debug info
        """
        square_diff=0
        # convert x to inner_p
        angle_inds = int(math.factorial(ndim) / math.factorial(2) / math.factorial(ndim - 2))
        metric_tensor, cocycle_rep = self.convert_params(x, ndim, angle_inds, int(order - 1),lattice_type,metric_tensor_std)
        if cocycle_rep is not None: 
            periodic_rep = np.concatenate((cycle_rep, cocycle_rep))
        else:
            periodic_rep=cycle_rep
        lattice_arcs = np.dot(cycle_cocycle_I, periodic_rep)
        inner_p = np.dot(np.dot(lattice_arcs, metric_tensor), lattice_arcs.T)
        # calculate distance matrix
        coordinates_temp=self.get_coordinates(lattice_arcs,num_nodes,shortest_path,spanning)
        coordinates_temp_cart=np.dot(coordinates_temp,lattice_vectors_scaled)  # using the orginal scaled lattice vectors to simplify calculation
        distance_matrix=self.all_distances(coordinates_temp_cart,coordinates_temp_cart)
        for k in range(len(colattice_inds[0])):
            i=colattice_inds[0][k]
            j=colattice_inds[1][k]
            if i==j:
                square_diff += (inner_p[i][j]-mat_target[i][j])**2 + 4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12  # inner_p单位是 Angstrom的平方
                print("bond:"+str(i)+','+str(j)+','+str(round(np.sqrt(inner_p[i][j]),2))+','+str(round(np.sqrt(mat_target[i][j]),2))+','+str(round((inner_p[i][j]-mat_target[i][j])**2,5))+','+str(round(4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12,5)))
            else:
                square_diff += angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2  # divide angle weight by 3
                print("angle:"+str(i)+','+str(j)+','+str(round(inner_p[i][j]/np.sqrt(inner_p[i][i])/np.sqrt(inner_p[j][j]),5))+','+str(round(mat_target[i][j]/np.sqrt(mat_target[i][i])/np.sqrt(mat_target[j][j]),5))+','+str(round(angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2,5)))
        # repulsive part of LJ potential to prevent atom collision
        if repul:
            for i in range(len(uncovered_pair)):  #  cutoff        # epsilon       # sigma\
                r_temp=distance_matrix[uncovered_pair[i][0],uncovered_pair[i][1]]
                if r_temp<uncovered_pair_lj[i][0]:
                    square_diff+=4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12 
                    print("repul:"+str(uncovered_pair[i][0])+','+str(uncovered_pair[i][1])+','+str(round(r_temp,4))+','+str(round(4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12,4)))
        return square_diff

    def to_structures(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.00,vbond_param_ave=0.01,repul=True):        
        """
        convert edge_indices, to_jimages and atom_types stored in the InvCryRep instance back to 
        3 pymatgen structure objects:
        (1) barycentric embedding net with rescaled lattice based on the average bond scaling factors
         calculated with modified GFN-FF predicted geometry 
        (2) non-barycentric net embedding that matches bond lengths and bond angles predicted with
         modified GFN-FF
        (3) non-barycentric net embedding undergone cell optimization using M3GNet IAPs
        if cell optimization failed, then output (1) and (2)
        """
        from datetime import datetime
        #print("before inner_p =", datetime.now())
        #print("after inner_p =", datetime.now())
        x_dat, net_voltage = self.convert_graph()
        #print(x_dat,net_voltage)
        net = Net(x_dat,dim=3)
        net.voltage = net_voltage
        if self.check_results:
            fig = plt.figure()
            nx.draw(net.graph, ax=fig.add_subplot(111))
            fig.savefig("graph.png")
        #print(net.graph.edges)
        inner_p_target, colattice_inds, colattice_weights = self.get_inner_p_target(bond_scaling)
        uncovered_pair = self.get_uncovered_pair(net.graph)
        uncovered_pair_lj = self.get_uncovered_pair_lj(uncovered_pair)
        covered_pair_lj = self.get_covered_pair_lj()
        net.simple_cycle_basis()
        net.get_lattice_basis()
        net.get_cocycle_basis()
        #deal with cocycle == none 
        if net.cocycle is not None: 
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
        else:
            net.periodic_rep=net.cycle_rep
        net.get_metric_tensor()
        lattice_vectors_std=np.linalg.cholesky(net.metric_tensor)
        arc_coord_std=net.lattice_arcs
        # get spanning and shortest_path_spanning_graph
        num_nodes=len(net.graph.nodes)
        G = nx.MultiDiGraph()
        G.add_nodes_from(net.vertices())
        G.add_edges_from(net.all_edges())
        edges=list(G.edges)
        G_nonDi = nx.MultiGraph()
        G_nonDi.add_nodes_from(net.vertices())
        G_nonDi.add_edges_from(net.all_edges())
        mst = tree.minimum_spanning_edges(G_nonDi, algorithm="kruskal", data=False) 
        spanning = list(mst)
         # convert spanning back to MultiDigraph's case
        for i in range(len(spanning)):   
            for j in range(len(edges)):
                if spanning[i][2]==edges[j][2]:
                    spanning[i]=edges[j]
        spanning_graph=nx.MultiGraph()
        spanning_graph.add_nodes_from(G.nodes)
        spanning_graph.add_edges_from(spanning)
        shortest_path = nx.shortest_path(spanning_graph, source='1')
        # calculate anisotropic scaled lattice vectors of standard placement
        inner_p_std = np.dot(np.dot(net.lattice_arcs, net.metric_tensor), net.lattice_arcs.T)
        lattice_vectors_scaled = self.get_rescaled_lattice_vectors(inner_p_target,inner_p_std,lattice_vectors_std,arc_coord_std)
        metric_tensor_std=np.dot(lattice_vectors_scaled,lattice_vectors_scaled.T)
        # add random pertubation to cocycle_rep to deal with unstable graph
        if self.unstable_graph: 
            net.cocycle_rep = np.zeros((net.order-1, net.ndim)) + .5
            #net.cocycle_rep = np.random.random((net.order-1, net.ndim)) - .5
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
            arc_coord_std=net.lattice_arcs
        # get the fractional coordinates of vertices of standard placement
        coordinates_std=self.get_coordinates(arc_coord_std,num_nodes,shortest_path,spanning) 
        # get the gfnff-scaled standard placement
        atom_symbols=[str(ElementBase.from_Z(i)) for i in self.atom_types]
        structure_recreated_std = Structure(lattice_vectors_scaled, atom_symbols,coordinates_std)
        # optimize X (lattice vectors and cocycle_rep)
        # opt x
        #print("before optimize =", datetime.now())
        # get lattice type
        lattice_length_list=[metric_tensor_std[0,0],metric_tensor_std[1,1],metric_tensor_std[2,2]]
        lattice_length_list_unique=list(set(lattice_length_list))
        lattice_type=len(lattice_length_list_unique)
        if lattice_type==2:
            if metric_tensor_std[0,0]==metric_tensor_std[1,1]:
                lattice_type=23
            if metric_tensor_std[0,0]==metric_tensor_std[2,2]:
                lattice_type=22
            if metric_tensor_std[2,2]==metric_tensor_std[1,1]:
                lattice_type=21
        x,bounds = self.initialize_x_bounds(net.ndim,net.cocycle_rep,metric_tensor_std,lattice_type,delta_theta,delta_x,lattice_expand,lattice_shrink)
        x=fmin_l_bfgs_b(self.func, x, fprime=None, args= \
        (net.ndim,net.order,inner_p_target,colattice_inds,colattice_weights,net.cycle_rep,net.cycle_cocycle_I, \
        num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
        lattice_vectors_scaled,atom_symbols,angle_weight,repul,lattice_type,metric_tensor_std), \
        approx_grad=True, bounds=bounds, m=10, factr=10000000.0, pgtol=1e-05, \
        epsilon=1e-08, iprint=- 1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
        #get optimized structure
        angle_inds = int(math.factorial(net.ndim) / math.factorial(2) / math.factorial(net.ndim - 2))
        net.metric_tensor, net.cocycle_rep = self.convert_params(x[0], net.ndim, angle_inds, int(net.order - 1),lattice_type,metric_tensor_std)
        lattice_vectors_new=np.linalg.cholesky(net.metric_tensor)
        if net.cocycle is not None: 
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
        else:
            net.periodic_rep=net.cycle_rep
        arc_coord_new=net.lattice_arcs
        coordinates_new=self.get_coordinates(arc_coord_new,num_nodes,shortest_path,spanning) 
        structure_recreated_opt = Structure(lattice_vectors_new, atom_symbols,coordinates_new)
        #print("after opt =", datetime.now())
        if self.check_results:
            print(x[0])
            print(self.func_check(x[0],net.ndim,net.order,inner_p_target,colattice_inds,colattice_weights,net.cycle_rep,net.cycle_cocycle_I, \
            num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
            lattice_vectors_scaled,atom_symbols,angle_weight,repul,lattice_type,metric_tensor_std))
        try:
            structure_recreated_opt2, final_energy_per_atom=self.m3gnet_relax(structure_recreated_opt)
            return [structure_recreated_std, structure_recreated_opt,  structure_recreated_opt2 ],final_energy_per_atom
        except:
            return [structure_recreated_std, structure_recreated_opt],0

    def SLICES2structure(self,SLICES):
        """
        convert a SLICES string back to its original crystal structure
        """
        self.from_SLICES(SLICES)
        structures,final_energy_per_atom = self.to_structures()
        return structures[-1],final_energy_per_atom


    def to_structure(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.000,vbond_param_ave=0.01,repul=True):
        """
        convert edge_indices, to_jimages and atom_types stored in the InvCryRep instance back to 
        a pymatgen structure object: non-barycentric net embedding undergone cell optimization 
        using M3GNet IAPs
        if cell optimization failed, then output non-barycentric net embedding that matches 
        bond lengths and bond angles predicted with modified GFN-FF
        """
        structures,final_energy_per_atom=self.to_structures(bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        return structures[-1]
        
    def to_relaxed_structure(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.000,vbond_param_ave=0.01,repul=True):
        """
        convert edge_indices, to_jimages and atom_types stored in the InvCryRep instance back to 
        a pymatgen structure object: non-barycentric net embedding undergone cell optimization 
        using M3GNet IAPs
        if cell optimization failed, then raise error
        """
        structures,final_energy_per_atom=self.to_structures(bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        if len(structures)==3:        
            return structures[-1],final_energy_per_atom
        else:
            raise Exception("relax failed")

    def to_4structures(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.000,vbond_param_ave=0.01,repul=True):
        """
        designed for benchmark 
        """
        structures,final_energy_per_atom=self.to_structures(bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        try:
            std2, final_energy_per_atom=self.m3gnet_relax(structures[0])
            return [structures[0], std2,  structures[1],structures[2] ],final_energy_per_atom
        except:
            return structures,final_energy_per_atom

    @function_timeout(seconds=60)
    def m3gnet_relax(self,struc):
        """
        cell optimization using M3GNet IAPs (time limit is set to 200 seconds 
        to prevent buggy cell optimization that takes fovever to finish)
        """
        relax_results = self.relaxer.relax(struc,fmax=self.fmax,steps=self.steps)
        final_structure = relax_results['final_structure']
        final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(struc))
        return final_structure,final_energy_per_atom


    def match_check(self,ori,opt,std,ltol=0.2, stol=0.3, angle_tol=5):
        """
        calculate match rates of structure (2) and (1) with respect to the original structure
        calculate topological distances of structuregraph of structure (2) and (1) with respect to 
        structuregraph of the original structure
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])

    def match_check2(self,ori,opt,std,ltol=0.2, stol=0.3, angle_tol=5):
        """
        calculate match rates of structure (2) and (1) with respect to the original structure
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std_checked)))

    def match_check3(self,ori,opt2,opt,std,ltol=0.2, stol=0.3, angle_tol=5):
        """
        calculate match rates of structure (3), (2) and (1) with respect to the original structure
        calculate topological distances of structuregraph of structure (3), (2) and (1) with respect to 
        structuregraph of the original structure
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt2_checked=Structure.from_str(opt2.to(fmt="poscar"),"poscar")
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_opt2=self.structure2structure_graph(opt2_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt2_checked))),str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt2,strict=False)["dist"]),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])

    def match_check4(self,ori,opt2,opt,std2,std,ltol=0.2, stol=0.3, angle_tol=5):
        """
        structure (4): barycentric embedding net with rescaled lattice undergone cell optimization using M3GNet IAPs
        calculate match rates of structure (3), (2), (4) and (1) with respect to the original structure
        calculate topological distances of structuregraph of structure (3), (2), (4) and (1) with respect to 
        structuregraph of the original structure
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt2_checked=Structure.from_str(opt2.to(fmt="poscar"),"poscar")
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        std2_checked=Structure.from_str(std2.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_opt2=self.structure2structure_graph(opt2_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sg_std2=self.structure2structure_graph(std2_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt2_checked))),str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std2_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt2,strict=False)["dist"]),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std2,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])
