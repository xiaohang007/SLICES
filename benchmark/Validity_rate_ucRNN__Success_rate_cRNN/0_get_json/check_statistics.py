# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json

with open('cifs_filtered.json', 'r') as f:
    cifs=json.load(f)
cifs_filtered=[]
sci_text=''

for i  in range(len(cifs)):
    eform=cifs[i]["formation_energy_per_atom"]
    sci_text+=str(eform)+'\n'

with open("eform_ori.csv","w") as f:
    f.write(sci_text)



