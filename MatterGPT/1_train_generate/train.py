# train_nprops.py
# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com

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

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SliceDataset
import math, os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    # 统一使用 prop_column_index_list，而不是单个的 prop_column_index
    parser.add_argument('--prop_column_index_list', type=int, nargs='*', default=[], 
                        help='List of indices for property columns. e.g. --prop_column_index_list 1 2 (for 2 properties)')
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="learning rate", required=False)
    
    parser.add_argument('--train_dataset', type=str, default="../0_dataset/train_data.csv", 
                        help="path to the training dataset file")
    parser.add_argument('--val_dataset', type=str, default="../0_dataset/val_data.csv", 
                        help="path to the validation dataset file")
    parser.add_argument('--slices_column_index', type=int, default=0, 
                        help='Index of the column for slices.')
    
    args = parser.parse_args()

    set_seed(42)

    # 1. 读取训练数据
    data_train = pd.read_csv(args.train_dataset, delimiter=',', header=0)
    slices_train = data_train.iloc[:, args.slices_column_index].tolist()
    
    # props_train 读取
    if len(args.prop_column_index_list) > 0:
        props_train = [
            list(map(float, row))
            for row in zip(*[data_train.iloc[:, idx].tolist() for idx in args.prop_column_index_list])
        ]
    else:
        # 如果没有传入性质列，就认为没性质
        props_train = None

    # 2. 读取验证数据（可选）
    slices_val = []
    props_val = []
    if args.val_dataset and os.path.exists(args.val_dataset):
        data_val = pd.read_csv(args.val_dataset, delimiter=',', header=0)
        slices_val = data_val.iloc[:, args.slices_column_index].tolist()
        
        if len(args.prop_column_index_list) > 0:
            props_val = [
                list(map(float, row))
                for row in zip(*[data_val.iloc[:, idx].tolist() for idx in args.prop_column_index_list])
            ]
        else:
            props_val = None

    # 打印一些检查信息
    print("First training slice:", slices_train[0])
    if props_train is not None:
        print("First training property row:", props_train[0])
        num_props = len(props_train[0])  # n_props
    else:
        print("No property columns found.")
        num_props = 0

    # 3. 构建 vocabulary
    slices_train = [sli.strip() for sli in slices_train]
    slices_val = [sli.strip() for sli in slices_val]
    lens = [len(sli.split(' ')) for sli in (slices_train + slices_val)]
    max_len = max(lens) if len(lens) > 0 else 0
    print('Max length of slices: ', max_len)
    
    voc_chars = construct_vocabulary(slices_train + slices_val)
    char_list = sorted(list(voc_chars) + ['<','>'])
    vocab_size = len(char_list)
    print("vocab_size: {}\n".format(vocab_size))
    print(char_list)

    # 4. 准备数据集
    train_dataset = SliceDataset(args, slices_train, char_list, max_len, props_train)
    val_dataset = None
    if len(slices_val) > 0:
        val_dataset = SliceDataset(args, slices_val, char_list, max_len, props_val)

    # 5. 设置模型
    mconf = GPTConfig(
        vocab_size=vocab_size, 
        block_size=max_len, 
        num_props=num_props,
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd,
    )
    model = GPT(mconf)

    # 6. 打印模型可训练参数
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The number of trainable parameters is: {}".format(params))

    # 7. 设置训练配置并开始训练
    tconf = TrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=0.1 * len(slices_train) * max_len,
        final_tokens=args.max_epochs * len(slices_train) * max_len,
        num_workers=4,
        ckpt_path=f'./model/{args.run_name}.pt' if args.run_name else None,
        block_size=max_len,
        generate=False
    )
    
    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        config=tconf,
        stoi=train_dataset.stoi,
        itos=train_dataset.itos,
        num_props=num_props,
    )

    df = trainer.train()
    # df 里可输出 loss 等信息  之前把val set和 val set弄混了，现在把这个代码里面的 val 相关的词汇都替换为val，除了函数里面的，多谢 
