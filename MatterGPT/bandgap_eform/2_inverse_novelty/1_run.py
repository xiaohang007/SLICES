import os
import glob
from slices.utils import splitRun_csv, show_progress, collect_csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
import pickle,json
from pymatgen.core.structure import Structure
def load_and_save_structure_database():
    with open('../../../data/mp20_nonmetal/cifs_filtered.json', 'r') as f:
        cifs = json.load(f)
    
    structure_database = []
    for i in range(len(cifs)):
        cif_string = cifs[i]["cif"]
        try:
            stru = Structure.from_str(cif_string, "cif")
            structure_database.append([stru, cifs[i]["band_gap"]])
        except Exception as e:
            print(e)
    
    # Serialize the data
    with open('structure_database.pkl', 'wb') as f:
        pickle.dump(structure_database, f)
        
def process_data():
    load_and_save_structure_database()
    splitRun_csv(filename='../1_train_generate/inverse_designed_2props_SLICES.csv', threads=10, skip_header=True)
    show_progress()
    collect_csv(output="results.csv",
                glob_target="./job_*/result.csv", cleanup=True,
                header="bandgap_target,eform_target,SLICES,poscar,novelty\n")


if __name__ == "__main__":
    process_data()
    
