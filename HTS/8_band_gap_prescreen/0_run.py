# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from slices.utils import temporaryWorkingDirectory,splitRun_csv,show_progress,collect_csv_filter
splitRun_csv(filename='../7_EAH_prescreen/results_7_EAH_prescreen_filtered.csv',threads=4,skip_header=True)
show_progress()
band_gap_lower_limit=0.1 # eV
collect_csv_filter(output="results_8_band_gap_prescreen.csv", glob_target="job_*/result.csv",\
            header='index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,dissimilarity,energy_above_hull_prescreen,band_gap_prescreen\n', \
            condition=lambda i: float(i.split(',')[-1]) > band_gap_lower_limit,cleanup=True)