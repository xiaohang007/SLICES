import pandas as pd
import argparse
from utils import *
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import MatterGPT, MatterGPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SliceDataset

from sklearn.model_selection import train_test_split


import math
import os
import configparser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--sym_dim', type=int, default=7,
                        help="dimension of symmetry embedding", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=5e-5, help="learning rate", required=False)
    parser.add_argument('--slices_column_index', type=int, default=0,
                        help='Index of the column for slices.')
    parser.add_argument('--prop_column_index_list', type=int, nargs='*', default=[],
                        help='List of indices for property columns. e.g. --prop_column_index_list 1 2')
    parser.add_argument('--train_dataset', type=str, default="../0_dataset/train_data.csv",
                        help="path to the training dataset file")
    parser.add_argument('--val_dataset', type=str, default="../0_dataset/val_data.csv",
                        help="path to the validation dataset file")

    args = parser.parse_args()

    set_seed(42)

    train_dataset_file = args.train_dataset
    data = pd.read_csv(train_dataset_file, delimiter=',')

    # 确保数据中包含必要的列
    if 'crystal_system' not in data.columns:
        raise ValueError("Dataset must contain 'crystal_system' column")
    if len(args.prop_column_index_list) == 0:
        raise ValueError("Property columns must be specified")

    slices_list = data.iloc[:, args.slices_column_index].tolist()
    props_list = [
        [float(x) for x in row]
        for row in zip(*[data.iloc[:, idx].tolist() for idx in args.prop_column_index_list])
    ]
    sym_list = data['crystal_system'].tolist()

    if args.val_dataset is not None:
        data_val = pd.read_csv(args.val_dataset, delimiter=',')
        val_slices = data_val.iloc[:, args.slices_column_index].tolist()
        val_props = [
            [float(x) for x in row]
            for row in zip(*[data_val.iloc[:, idx].tolist() for idx in args.prop_column_index_list])
        ]
        val_sym = data_val['crystal_system'].tolist()

        train_slices = slices_list
        train_props = props_list
        train_sym = sym_list
    else:
        train_slices, val_slices, train_props, val_props, train_sym, val_sym = train_val_split(
            slices_list, props_list, sym_list, val_size=0.1, random_state=1234)

    print("First training slice:", train_slices[0])
    if train_props is not None:
        print("First training property row:", train_props[0])
        num_props = len(train_props[0])
    else:
        print("No property columns found.")
        num_props = 0

    print('Max length of slices: ', max([len(sli.split(' ')) for sli in train_slices + val_slices]))
    
    # Calculate max_len once and reuse it
    max_len = max([len(sli.split(' ')) for sli in train_slices + val_slices])
    
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(train_slices + val_slices)
    char_list = sorted(list(voc_chars) + ['<','>'])
    vocab_size = len(char_list)
    print("vocab_size: {}\n".format(vocab_size))
    print(char_list)
    print(len(char_list))

    train_datasets = SliceDataset(args, train_slices, char_list, max_len,
                                   prop_list=train_props, sym_list=train_sym)
    val_datasets = SliceDataset(args, val_slices, char_list, max_len,
                                  prop_list=val_props, sym_list=val_sym)
    
    mconf = MatterGPTConfig(vocab_size, max_len, 
                      num_props=num_props,
                      sym_dim=args.sym_dim,
                      n_layer=args.n_layer, 
                      n_head=args.n_head, 
                      n_embd=args.n_embd)
                                                
    model = MatterGPT(mconf)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # 在训练前或训练结束时，保存配置和词汇表
    if args.run_name:
        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存 vocabulary
        vocab_file = os.path.join(model_dir, f'{args.run_name}_vocab.json')
        vocab_data = {
            'stoi': {ch: i for i, ch in enumerate(char_list)},
            'itos': {str(i): ch for i, ch in enumerate(char_list)}
        }
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # 保存模型参数配置
        config = configparser.ConfigParser()
        config['ModelParameters'] = {
            'vocab_size': str(vocab_size),
            'block_size': str(max_len),
            'num_props': str(num_props),
            'n_layer': str(args.n_layer),
            'n_head': str(args.n_head),
            'n_embd': str(args.n_embd),
            'sym_dim': str(args.sym_dim)
        }
        config_file = os.path.join(model_dir, f'{args.run_name}.ini')
        with open(config_file, 'w') as f:
            config.write(f)

    print("Model architecture:")
    print(model)
    print("The number of trainable parameters is: {}".format(params))
    
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(slices_list)*max_len,
                            final_tokens=args.max_epochs*len(slices_list)*max_len,
                            num_workers=10, 
                            ckpt_path=f'./model/{args.run_name}.pt',
                            block_size=max_len, generate=False)

    trainer = Trainer(model, train_datasets, val_datasets,
                        tconf, train_datasets.stoi, train_datasets.itos)

    df = trainer.train()




