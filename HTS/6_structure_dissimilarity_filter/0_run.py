# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from invcryrep.utils import temporaryWorkingDirectory,splitRun,splitRun_csv,show_progress,collect_csv_filter
import os,sys,glob
from monty.serialization import loadfn, dumpfn
# save structural fingerprints to json
with temporaryWorkingDirectory("./0_save_structure_fingerprint"):
    splitRun(filename='../../0_get_json_mp_api/prior_model_dataset_filtered.json',threads=16,skip_header=False)
    show_progress()
    training_fingerprint_list=[]
    for i in glob.glob("job_*/training_fingerprint_list.json.gz"):
        training_fingerprint_list+=loadfn(i)
    dumpfn(training_fingerprint_list, "../training_fingerprint_list.json.gz") # or .json.gz
    print("../training_fingerprint_list.json.gz has been generated")
    for i in glob.glob("job_*"):
        os.system("rm -r "+i)
# Rule out crystals displaying minimum structural dissimilarity value < 0.75
splitRun_csv(filename='../5_symmetry_filter_refine/results_5_symmetry_filter_refine_filtered.csv',threads=16,skip_header=True)
show_progress()
# collect results
dissimilarity_limit=0.75
collect_csv_filter(output="results_6_structure_dissimilarity_filter.csv", glob_target="job_*/result.csv",\
            header='index,SLICES,POSCAR,formula,energy_per_atom,energy_per_atom_sym,space_group_number,dissimilarity\n', \
            condition=lambda i: float(i.split(',')[-1]) >= dissimilarity_limit,cleanup=True)
# the lambda expression is the screening criteria