# -*- coding: utf-8 -*-
# Hang Xiao 2023.08
# xiaohang07@live.cn
import os,sys,json
import numpy as np
import pandas as pd
with open('../../0_get_json/0_get_mp20_json/cifs.json', 'r') as f: #_filtered
    cifs=json.load(f)
cifs_filtered=[]
sci_text=''

for i  in range(len(cifs)):
    eform=cifs[i]["formation_energy_per_atom"]
    sci_text+=str(eform)+'\n'

with open("eform_ori.csv","w") as f:
    f.write("formation_energy\n")
with open("eform_ori.csv","a") as f:
    f.write(sci_text)




data = pd.read_csv('eform_ori.csv')
os.system("rm eform_ori.csv")


percentiles = [5, 10, 15]
minimal_percentiles = [data['formation_energy'].quantile(p / 100) for p in percentiles]


for p, value in zip(percentiles, minimal_percentiles):
    print(f"The {p}% percentile of formation energy is: {value}")
    

input_file = "../5_eform_PBE/results_5_eform_PBE.csv"


df = pd.read_csv(input_file)

percentiles_mp20 =minimal_percentiles
eform_PBE=list(df["eform_PBE"].values)
num_materials=len(eform_PBE)
print(num_materials)
sr_5=0
sr_10=0
sr_15=0
for i in range(len(eform_PBE)):
    if eform_PBE[i]<=percentiles_mp20[2]:
        sr_15+=1
    if eform_PBE[i]<=percentiles_mp20[1]:
        sr_10+=1
    if eform_PBE[i]<=percentiles_mp20[0]:
        sr_5+=1

print({'SR5': sr_5/num_materials, 'SR10': sr_10/num_materials, 'SR15': sr_15/num_materials})

