# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
import configparser
from sklearn.model_selection import ParameterGrid
config = configparser.ConfigParser()
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
   os.chdir('job_'+str(i))
   os.system('python 1_splitRun.py')
   os.chdir('..')

   
