# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from slices.utils import temporaryWorkingDirectory,splitRun_csv,show_progress,collect_csv_filter
import os
# download relevant entries for high-throughput energy above hull calculation
os.system("rm result.csv > /dev/null 2>&1")
os.system("python ehull.py -i '../6_structure_dissimilarity_filter/results_6_structure_dissimilarity_filter_filtered.csv' -o result.csv")
print("competitive_compositions.json.gz has been generated")
# calculate structure_dissimilarity
splitRun_csv(filename='../6_structure_dissimilarity_filter/results_6_structure_dissimilarity_filter_filtered.csv',threads=16,skip_header=True)
show_progress()
# collect results
ehull_limit=0.5 # eV/atom
collect_csv_filter(output="results_7_EAH_prescreen.csv", glob_target="job_*/result.csv",\
            header='index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,dissimilarity,energy_above_hull_prescreen\n', \
            condition=lambda i: float(i.split(',')[-1]) <= ehull_limit,cleanup=True)