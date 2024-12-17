# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import pandas as pd
import argparse,os
from utils import *
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SliceDataset
from sklearn.model_selection import train_test_split
import math

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=5e-5, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--train_dataset', type=str, default="../../../data/mp20_nonmetal/train_data_reduce_zero.csv",
                        help="path to the training dataset file")
    parser.add_argument('--test_dataset', type=str, default="../../../data/mp20_nonmetal/test_data_reduce_zero.csv",
                        help="path to the test dataset file (optional)")
    args = parser.parse_args()

    set_seed(42)

    # train data
    train_dataset_file = args.train_dataset
    data_train = pd.read_csv(train_dataset_file, delimiter=',',header=0)
    slices_train = data_train.iloc[:, 0].tolist()
    props_train = data_train.iloc[:, 1].tolist() 
    slices_test=[]

    # test data (optional)    
    test_dataset_file = args.test_dataset
    if os.path.exists(test_dataset_file):
        data_test = pd.read_csv(test_dataset_file, delimiter=',',header=0)
        slices_test = data_test.iloc[:, 0].tolist()
        props_test = data_test.iloc[:, 1].tolist()
    
    print(slices_train[0])
    print(props_train[0])
    print("Constructing vocabulary...")
    slices_train = [sli.strip() for sli in (slices_train)]
    slices_test = [sli.strip() for sli in (slices_test)]
    lens = [len(sli.strip().split(' ')) for sli in (slices_train+slices_test)]
    max_len = max(lens)
    print('Max length of slices: ', max_len)   
    voc_chars = construct_vocabulary(slices_train+slices_test)
    char_list = sorted(list(voc_chars)+['<','>'])
    vocab_size = len(char_list)
    print("vocab_size: {}\n".format(vocab_size))
    print(char_list)
    print(len(char_list))

    # prepare datasets
    train_datasets = SliceDataset(args, slices_train, char_list, max_len, props_train)
    if len(slices_test) >0: 
        test_datasets = SliceDataset(args, slices_test, char_list, max_len, props_test)
    
    mconf = GPTConfig(vocab_size, max_len, num_props=args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                        lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    params = sum([np.prod(p.size()) for p in model_parameters])
    #print("Model architecture:")
    #print(model)
    print("The number of trainable parameters is: {}".format(params))

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(slices_train)*max_len,
                            final_tokens=args.max_epochs*len(slices_train)*max_len,num_workers=4, 
                            ckpt_path=f'./model/{args.run_name}.pt', block_size=max_len, generate=False)
    if len(slices_test) ==0: 
        trainer = Trainer(model, train_datasets, None, tconf, train_datasets.stoi, train_datasets.itos,args.num_props,None)
    else:
        trainer = Trainer(model, train_datasets, test_datasets, tconf, train_datasets.stoi, train_datasets.itos,args.num_props,None)

    df = trainer.train()



