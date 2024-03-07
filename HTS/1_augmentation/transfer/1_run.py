# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob
import re
import numpy as np
import math,json
from invcryrep.utils import splitRun,show_progress,collect_json,collect_csv
splitRun(filename='../../0_get_json_mp_api/transfer_model_dataset_filtered.json',threads=8,skip_header=False)
show_progress()
collect_csv(output="../transfer_aug.sli", \
    glob_target="./**/result.sli",header="",cleanup=True)
