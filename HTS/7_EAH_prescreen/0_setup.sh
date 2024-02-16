#/bin/bash

rm result.csv
python ehull.py -i "../5_symmetry_filter_refine/results_5_symmetry_filter_refine_filtered.csv" -o result.csv
