# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import configparser
from sklearn.model_selection import ParameterGrid
from slices.utils import splitRun,show_progress,collect_json,collect_csv_filter,temporaryWorkingDirectory
config = configparser.ConfigParser()
import os,sys,glob,json
import pandas as pd
import numpy as np

#bond_scaling=
param_grid = {
'bond_scaling': [1.05],  #1.0,1.1,1.2,1.3
'delta_theta': [0.005],
'delta_x':[0.45],
'lattice_shrink':[1],
'lattice_expand':[1.25],
'angle_weight':[0.5],
'vbond_param_ave_covered':[0],
'vbond_param_ave':[0.01],
'repul':[True],
'graph_method':['crystalnn'],
}
grid=list(ParameterGrid(param_grid))
os.system('rm -rf job_*')
for i in range(len(grid)):
    os.mkdir('job_'+str(i))
    os.system('cp -r ./template/. job_'+str(i))
    config["Settings"] = {'bond_scaling':grid[i]['bond_scaling'] ,'delta_theta':grid[i]['delta_theta'],'delta_x':grid[i]['delta_x'],
    'lattice_shrink':grid[i]['lattice_shrink'],'lattice_expand':grid[i]['lattice_expand'],'angle_weight':grid[i]['angle_weight'],
    'vbond_param_ave_covered':grid[i]['vbond_param_ave_covered'],'vbond_param_ave':grid[i]['vbond_param_ave'],'repul':grid[i]['repul'],'graph_method':grid[i]['graph_method']}
    with open('./job_'+str(i)+'/settings.ini', 'w') as configfile:
        config.write(configfile)
    with temporaryWorkingDirectory('job_'+str(i)):
        splitRun(filename='../../../data/mp20/cifs_filtered.json',threads=16,skip_header=False)
        show_progress()
        collect_csv_filter(output="results.csv",condition=lambda i: len(i.split(','))==12, glob_target="job_*/result.csv",header="'name,opt2_match,opt_match,std_match,opt2_match2,opt_match2,std_match2,opt2_topo2,opt_topo2,std_topo2,natoms,time\n",cleanup=True)
        

# collect results
os.system("rm *.csv")
with open("benchmark_result.csv",'w')as f:
    f.write('folder,opt2_match_sum,opt_match_sum,std_match_sum,opt2_match2_sum,opt_match2_sum,std_match2_sum,opt2_topo2_sum,opt_topo2_sum,std_topo2_sum,time_sum, \
    bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul,graph_method\n')

with open('../../data/mp20/cifs_filtered.json', 'r') as f:
    cifs=json.load(f)
num=len(cifs)
config = configparser.ConfigParser()
for i in glob.glob("job_*/results_filtered.csv"):  # filtered result is result_filtered.csv
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
    with open("benchmark_result.csv",'a')as r:
        r.write(i.split('/')[0]+','+str(data['opt2_match'].sum()/num*100)+','+str(data['opt_match'].sum()/num*100)+','+str(data['std_match'].sum()/num*100)+ \
        ','+str(data['opt2_match2'].sum()/num*100)+','+str(data['opt_match2'].sum()/num*100)+','+str(data['std_match2'].sum()/num*100)+ \
        ','+str(data['opt2_topo2'].sum()/num)+','+str(data['opt_topo2'].sum()/num)+','+str(data['std_topo2'].sum()/num)+','+str(data['time'].sum()/20/3600)+','+ \
        str(bond_scaling)+','+str(delta_theta)+','+str(delta_x)+','+str(lattice_shrink)+','+str(lattice_expand)+','+str(angle_weight)+','+str(vbond_param_ave_covered)+','+str(vbond_param_ave)+','+str(repul)+','+str(graph_method)+'\n')


   
