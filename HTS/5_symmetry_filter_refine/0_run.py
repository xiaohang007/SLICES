# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from slices.utils import temporaryWorkingDirectory,splitRun_csv,show_progress,collect_csv
import pandas as pd
splitRun_csv(filename='../4_composition_filter/results_4_composition_filter.csv',threads=16,skip_header=True)
show_progress()
collect_csv(output="results_5_symmetry_filter_refine.csv", glob_target="job_*/result.csv",\
            header='index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number\n',index=True,cleanup=True)
df = pd.read_csv("results_5_symmetry_filter_refine.csv")
result = df.loc[df['space_group_number'] != 1].groupby(['formula','space_group_number'], group_keys=False).apply(lambda x: x[x.energy_per_atom_sym==x.energy_per_atom_sym.min()])
result.to_csv("results_5_symmetry_filter_refine_filtered.csv", index=False)