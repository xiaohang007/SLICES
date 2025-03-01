from utils import *
from dataset import SliceDataset

import math
from tqdm import tqdm
import argparse,ast
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from slices import check_SLICES, get_canonical_SLICES


slices="+ t a ODD a DOD b OOO c OOO d OOO e OOO o Ca Pm Hg Hg 0 2 ooo 0 2 oo+ 0 2 +oo 0 2 o+o 0 1 oo+ 0 1 o+o 0 1 o++ 0 1 +oo 0 1 +o+ 0 1 ++o 0 3 -oo 0 3 o-o 0 3 oo- 0 3 ooo 1 3 --- 1 3 --o 1 3 o-- 1 3 -o- 1 2 -oo 1 2 o-o 1 2 oo- 1 2 ooo 2 3 --o 2 3 -o- 2 3 -oo 2 3 o-- 2 3 o-o 2 3 oo-  "
# Canonicalization and deduplication

print(check_SLICES(slices))

