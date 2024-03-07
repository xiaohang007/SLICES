# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from invcryrep.utils import temporaryWorkingDirectory,splitRun_csv,show_progress,collect_csv

splitRun_csv(filename='../3_inverse/results_3_inverse.csv',threads=16,skip_header=True)
show_progress()
collect_csv(output="results_4_composition_filter.csv", glob_target="job_*/result.csv",\
            header='SLICES,formula,energy_per_atom,POSCAR\n',cleanup=True)