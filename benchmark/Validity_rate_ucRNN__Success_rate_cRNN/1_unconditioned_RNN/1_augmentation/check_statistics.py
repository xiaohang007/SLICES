# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,json

with open('prior_aug_50x.sci', 'r') as f:
    cifs=f.readlines()
cifs_filtered=[]
sci_text=''

for i  in range(len(cifs)):
    eform=cifs[i].split(",")[1]
    sci_text+=str(eform.strip())+'\n'

with open("eform_aug50.csv","w") as f:
    f.write(sci_text)



