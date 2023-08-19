#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,glob,json
import pandas as pd
import numpy as np

pwd=os.getcwd()

os.system("rm *.csv")
with open("results_collection_"+pwd.split("/")[-1]+".csv",'w')as f:
    f.write('folder,opt2_match_sum,opt_match_sum,std_match_sum,opt2_match2_sum,opt_match2_sum,std_match2_sum,opt2_topo2_sum,opt_topo2_sum,std_topo2_sum,time_sum, \
    bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul,graph_method\n')

for i in glob.glob("job_*"):
    os.chdir(i)
    os.system("python 2_collect_clean_glob_details.py")
    os.chdir("..")

with open('/crystal/benchmark/get_json/cifs_filtered.json', 'r') as f:
    cifs=json.load(f)
num=len(cifs)
import configparser
config = configparser.ConfigParser()
for i in glob.glob("job_*/result*.csv"):
    config.read('./'+i.split('/')[-2]+'/settings.ini') #path of your .ini file
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
    data = pd.read_csv(i,dtype={'opt2_match': np.int32,'opt_match': np.int32,'std_match': np.int32,'opt2_match2': np.int32,'opt_match2': np.int32,'std_match2': np.int32,'opt2_topo2': np.float64, 'opt_topo2': np.float64,'std_topo2':np.float64, 'time':np.float64})
    
#    with open("results_collection.csv",'a')as r:
#        r.write(i.split('/')[0]+','+str(pd.DataFrame(data['recreated_match']).sum(numeric_only=True))+'\n')


    with open("results_collection_"+pwd.split("/")[-1]+".csv",'a')as r:
        r.write(i.split('/')[0]+','+str(data['opt2_match'].sum()/num*100)+','+str(data['opt_match'].sum()/num*100)+','+str(data['std_match'].sum()/num*100)+ \
        ','+str(data['opt2_match2'].sum()/num*100)+','+str(data['opt_match2'].sum()/num*100)+','+str(data['std_match2'].sum()/num*100)+ \
        ','+str(data['opt2_topo2'].sum()/num)+','+str(data['opt_topo2'].sum()/num)+','+str(data['std_topo2'].sum()/num)+','+str(data['time'].sum()/20/3600)+','+ \
        str(bond_scaling)+','+str(delta_theta)+','+str(delta_x)+','+str(lattice_shrink)+','+str(lattice_expand)+','+str(angle_weight)+','+str(vbond_param_ave_covered)+','+str(vbond_param_ave)+','+str(repul)+','+str(graph_method)+'\n')


#    with open("results_collection.csv",'a')as r:
#        r.write(i.split('/')[0]+','+str(pd.DataFrame(data['recreated_match']).sum(axis=0,numeric_only=True))+','+str(pd.DataFrame(data['recreated_std_match']).sum(axis=0,numeric_only=True))+ \
#        ','+str(pd.DataFrame(data['recreated_topo']).sum(axis=0,numeric_only=True))+','+str(pd.DataFrame(data['recreated_std_topo']).sum(axis=0,numeric_only=True))+'\n')
                
       
#print(data['recreated_match'].sum(axis=0))
#print(pd.DataFrame(data['recreated_match']))

#for i in glob.glob("job_*"):
#    os.system("rm -r "+i)
                
                
#    with open("results_collection.csv",'a')as r:
#        r.write(i.split('/')[0]+','+str(pd.DataFrame(data['recreated_match']).sum(axis=0,numeric_only=True))+','+str(pd.DataFrame(data['recreated_std_match']).sum(axis=0,numeric_only=True))+ \
#        ','+str(pd.DataFrame(data['recreated_topo']).sum(axis=0,numeric_only=True))+','+str(pd.DataFrame(data['recreated_std_topo']).sum(axis=0,numeric_only=True))+'\n')
                
                
