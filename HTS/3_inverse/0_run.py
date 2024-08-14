# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from slices.utils import temporaryWorkingDirectory,splitRun_csv,show_progress,collect_csv
splitRun_csv(filename='../2_train_sample/sampled.sli',threads=16,skip_header=False)
show_progress()
collect_csv(output="results_3_inverse.csv", glob_target="job_*/result.csv",\
            header='SLICES,formula,energy_per_atom,POSCAR\n',cleanup=True)