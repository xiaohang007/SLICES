# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,glob
import re
import numpy as np
import math,json
from invcryrep.utils import splitRun,show_progress,collect_json
splitRun(filename='../transfer_learning_dataset.json',threads=8,skip_header=False)
show_progress()
collect_json(output="../transfer_model_dataset_filtered.json", \
    glob_target="./**/output.json",cleanup=True)
