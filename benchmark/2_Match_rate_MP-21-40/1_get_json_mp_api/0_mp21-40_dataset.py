# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import json,os
from itertools import zip_longest
# An optional utility to display a progress bar
# for long-running loops. `pip install tqdm`.
from tqdm import tqdm
from mp_api.client import MPRester
import configparser
config = configparser.ConfigParser()
config.read('/crystal/APIKEY.ini') #path of your .ini file
apikey = config.get("Settings","API_KEY")
with MPRester(apikey) as mpr:
    docs = mpr.summary.search(formation_energy=(-10000,2),num_sites=(21,40),energy_above_hull=(0,0.08),fields=["material_id"])
    #mpid_bgap_dict = [{"material_id":doc.material_id,"cif": doc.structure.to("cif")}  for doc in docs]


oxide_mp_ids = [e.material_id for e in docs]
print(len(oxide_mp_ids))
# A utility function to "chunk" our queries

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

data = []
mpid_groups = [g for g in grouper(oxide_mp_ids, 1000)]
for group in tqdm(mpid_groups):
    # The last group may have fewer than 1000 actual ids,
    # so filter the `None`s out.
    #mpid_list = filter(None, group)
    temp=[]
    for i in group:
        if i != None:
            temp.append(i)
    docs = mpr.summary.search(material_ids=temp, fields=["material_id", "structure"])
    data.extend(docs)

dict_json = [{"material_id":str(e.material_id),"cif":e.structure.to(fmt="cif")} for e in data]

# exclude elements
from pymatgen.core.structure import Structure
exclude_elements=['Fr' , 'Ra','Ac','Th','Pa','U','Np',\
          'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf',\
          'Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',\
          'Lv','Ts','Og']
flitered_json = []
for i in dict_json:
    ori = Structure.from_str(i['cif'],fmt="cif")
    species=[str(j) for j in ori.species]
    flag=0
    for j in species:
        if j in exclude_elements:
            flag+=1
            break
    if not flag and i["material_id"] != None:
        flitered_json.append(i)
print(len(flitered_json))

with open('cifs.json', 'w') as f:
    json.dump(flitered_json, f)

"""
for d in tqdm(data):
    with open("mp_oxide_cifs/{}.cif".format(d.material_id), 'w') as f:
        f.write(d.structure.to(fmt="cif"))

"""
