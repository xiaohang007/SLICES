# -*- coding: UTF-8 -*-
# Hang Xiao 2023.07
# xiaohang07@live.cn
import numpy as np
from io import StringIO
import pandas as pd


# 3*3*3 supercell: 27 cells in -1, 0, 1 offsets in the x, y, z dimensions
OFFSET = np.array([
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
LJ_PARAMS_LIST = [
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
PERIODIC_DATA = pd.read_csv(
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