# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import json,os
from itertools import zip_longest
import pickle
# An optional utility to display a progress bar
# for long-running loops. `pip install tqdm`.
from tqdm import tqdm
from mp_api.client import MPRester

os.system('rm -rf mp_oxide_cifs')
if not os.path.exists('mp_oxide_cifs'):
    os.mkdir('mp_oxide_cifs')
with MPRester("40kzWCLsmGxC73szhK73Ym5aLfcYU1fK") as mpr:
    docs = mpr.summary.search(exclude_elements=['Fr' , 'Ra','Ac','Th','Pa','U','Np',\
          'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf',\
          'Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',\
          'Lv','Ts','Og'],num_sites=(21,1000),fields=["composition_reduced"])
    #mpid_bgap_dict = [{"material_id":doc.material_id,"cif": doc.structure.to("cif")}  for doc in docs]


composition_reduced_list = [e.composition_reduced for e in docs]
composition_reduced_list_unique=list(set(composition_reduced_list))

formula_reduced_list_unique = [str(e).replace(' ', '') for e in composition_reduced_list_unique] 


with open("formula_reduced_list_unique.pickle", "wb") as fp:   #Pickling
    pickle.dump(formula_reduced_list_unique, fp)
#with open('composition_reduced_list_unique.json', 'w') as f:
    #json.dump(composition_reduced_list_unique, f)

