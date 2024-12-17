from utils import *
from dataset import SliceDataset
from model import GPT, GPTConfig
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


results = pd.read_csv("band_gap_test_10x_stratified.csv")

# Canonicalization and deduplication
results['canonical_SLICES'] = results['SLICES'].apply(lambda x: get_canonical_SLICES(x, 4))
results.drop_duplicates(subset='canonical_SLICES', inplace=True)
results.drop('canonical_SLICES', axis=1, inplace=True)
print(len(results))

