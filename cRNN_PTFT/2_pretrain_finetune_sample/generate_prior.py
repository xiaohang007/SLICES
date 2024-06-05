#!/usr/bin/env python
import numpy,math
import torch,os
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate, unique
import torch.nn as nn
import argparse
import pandas as pd
import tempfile
import subprocess
import re
import gc
from slices import check_SLICES
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()


def sample_slices(voc_dir, gen_size,batch_size, outfn,tf_dir,append=False):
    """Sample smiles using the transferred model"""
    voc = Vocabulary(init_from_file=voc_dir)
    transfer_model = RNN(voc)
    if append:
        output = open(outfn, 'a')
    else:
        output = open(outfn, 'w')
    if not append:
        output.write('slices\n')
    if torch.cuda.is_available():
        transfer_model.rnn.load_state_dict(torch.load(tf_dir))
    else:
        transfer_model.rnn.load_state_dict(torch.load(tf_dir,map_location=lambda storage, loc:storage))


    gen_iter = math.ceil(gen_size / batch_size)
    valid = 0
    

    for i in tqdm(range(gen_iter)):    
        seqs, likelihood, _ = transfer_model.sample(int(batch_size))
        for i, seq in enumerate(seqs.cpu().numpy()):
            slices = voc.decode(seq)
            print(slices)
            if check_SLICES(slices,strategy=4,dupli_check=False,graph_rank_check=False):
                valid+=1
                output.write(slices+'\n')
    tqdm.write("\n{:>4.1f}% valid SLICES".format(100 * valid / gen_size))
    output.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain for SLICES generation")
    parser.add_argument('--voc', action='store',
                        default='data/Voc_withda', help='Directory for the vocabulary')
    parser.add_argument('--save_slices', action='store', default='cano_acceptors_smi.csv',
                        help='Directory of the SMILES file for tranfer learning')
    parser.add_argument('--model', action='store', default='data/Prior_gua_withda.ckpt',
                        help='Directory of the prior trained RNN')
    parser.add_argument('--gen_size', action='store', default='64', type=float,
                        help='Number of SMILES to sample for transfer learning')
    parser.add_argument('--batch_size', action='store', default='64',type=float,
                        help='Number of SMILES to sample for transfer learning')
    parser.add_argument("--scaler", required=False, default=0,  type=int, help="Scaler to use for regression. 0 for no scaling, 1 for min-max scaling, 2 for standard scaling. Default: 0")
    args = parser.parse_args()

    sample_slices(args.voc, args.gen_size,args.batch_size,args.save_slices,args.model, append=False)



