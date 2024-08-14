# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
from slices.utils import * splitRun,show_progress,collect_json,collect_csv
splitRun(filename='../prior_model_dataset.json',threads=8,skip_header=False)
show_progress()
collect_json(output="../prior_model_dataset_filtered.json", \
    glob_target="./**/output.json",cleanup=True)
