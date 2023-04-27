#/bin/bash
source /home/yanchen/miniconda3/etc/profile.d/conda.sh
conda activate alignn
rm result.csv
python ehull.py -i "results_5_symmetry_filter_refine_filtered.csv" -o result.csv
