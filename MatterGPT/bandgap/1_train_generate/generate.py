# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
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

def extract_model_params(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    state_dict = torch.load(model_path, map_location=torch.device(device))
    
    vocab_size = state_dict['tok_emb.weight'].size(0)
    block_size = state_dict['pos_emb'].size(1)
    n_embd = state_dict['tok_emb.weight'].size(1)
    n_layer = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('blocks.')]) + 1
    
    n_props = 0
    if 'prop_nn.weight' in state_dict:
        n_props = state_dict['prop_nn.weight'].size(1)
    
    return {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_props': n_props
    }

def parse_prop_condition(s):
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python expression
        parsed = ast.literal_eval(s)
        
        # Ensure the result is a list
        if not isinstance(parsed, list):
            raise ValueError("Input must be a list")
        
        # Convert all elements to float
        float_list = [float(x) for x in parsed]
        
        return float_list
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid prop_condition format: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
    parser.add_argument('--batch_size', type=int, default=128, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=10000, help="number of times to generate from a batch", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)
    parser.add_argument('--prop_targets', type=parse_prop_condition, required=True,
                    help='Property condition as a string representation of a list of target values, e.g., "[2.1, 3.5, 1.7, 4.0]"')

    args = parser.parse_args()

    model_path = './model/' + args.model_weight
    params = extract_model_params(model_path)
    nprops = params["n_props"]
    #for key, value in params.items():
        #print(f"{key}: {value}")
    #print("n_head: ", args.n_head)
    prop_condition = args.prop_targets # desired band gap

    for i in range(len(prop_condition)):
        print("Target "+str(i+1)+" is bandgap: "+str(prop_condition[i])+" eV.")
    #for key, value in params.items():
        #print(f"{key}: {value}")
    #print("n_head: ", args.n_head)
     
    char_list = sorted(set(read_vocab(fname="./Voc_prior")+['<','>']))
    stoi = {ch:i for i,ch in enumerate(char_list)}
    itos = {i:ch for i,ch in enumerate(char_list)}

    mconf = GPTConfig(params["vocab_size"], params["block_size"], num_props=nprops,
                   n_layer=params["n_layer"], n_head=args.n_head, n_embd=params["n_embd"],
                   lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading model')
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    print('Model loaded')

    gen_iter = math.ceil(args.gen_size / args.batch_size)
    
    all_slices = []
    model.eval()

    if prop_condition is not None:
        for c in prop_condition:
            for i in tqdm(range(gen_iter)):
                context = '>'
                x = torch.tensor([stoi[context]], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
                p = torch.tensor([[c]]).repeat(args.batch_size, 1).to(device) if nprops == 1 else torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to(device)

                y = model.sample(x, params["block_size"], temperature=0.9, do_sample=True, top_k=50, top_p=0.9, prop=p)  
                for gen_mol in y:
                    completion = " ".join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace("<", "").strip()
                    if check_SLICES(completion, 4):
                        all_slices.append({'bandgap': c, 'SLICES': completion})
                    else:
                        pass
                        #print(f"Invalid SLICES: {completion}")
                        
    results = pd.DataFrame(all_slices)
    print(f"Total generated SLICES: {len(results)}")
    # Canonicalization and deduplication
    results['canonical_SLICES'] = results['SLICES'].apply(lambda x: get_canonical_SLICES(x, 4))
    results.drop_duplicates(subset='canonical_SLICES', inplace=True)
    results.drop('canonical_SLICES', axis=1, inplace=True)

    print(f"Unique canonical SLICES: {len(results)}")

    # Save unique canonical SLICES with their target values
    results.to_csv(f'{args.csv_name}.csv', index=False)

    print(f"Valid ratio: {np.round(len(all_slices)/(args.batch_size*gen_iter*len(prop_condition)), 3)}")
    print(f"Unique ratio: {np.round(len(results)/len(all_slices), 3)}")
