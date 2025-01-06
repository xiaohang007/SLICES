# generate_nprops.py
# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com

import math
from tqdm import tqdm
import argparse, ast
import pandas as pd
import torch
import numpy as np
import os
import sys

from utils import *
from dataset import SliceDataset
from model import GPT, GPTConfig
from slices import check_SLICES, get_canonical_SLICES


def extract_model_params(model_path):
    """读取已有模型的权重信息（词表大小、block_size、n_layer、n_embd、n_props 等）"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # vocab_size
    vocab_size = state_dict['tok_emb.weight'].size(0)
    # block_size
    block_size = state_dict['pos_emb'].size(1)
    # n_embd
    n_embd = state_dict['tok_emb.weight'].size(1)
    
    # n_layer，需要找 blocks.X 中最大的 X
    n_layer = max([
        int(key.split('.')[1]) 
        for key in state_dict.keys() 
        if key.startswith('blocks.')
    ]) + 1

    # n_props
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
    """
    统一解析 --prop_targets 参数。
    让用户始终传一个 list of list 的字符串。
    比如 "[[1.0,2.0],[3.0,4.0]]" 或 "[[2.1],[3.5],[1.7],[4.0]]"。
    """
    try:
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, list) or not all(isinstance(row, list) for row in parsed):
            raise ValueError("prop_targets should be a list of list, e.g., [[1.0,2.0],[3.0,4.0]]")
        # 转成 float
        return [ [float(x) for x in row] for row in parsed ]
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid prop_condition format: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--output_csv', type=str, help="name to save the generated slices in csv format", required=True)
    parser.add_argument('--batch_size', type=int, default=128, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=10000, help="number of total generation attempts", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)
    
    parser.add_argument('--prop_targets', type=parse_prop_condition, required=True,
                        help='''Property condition as a string representation of a list of lists.
                        e.g., "[[1.0,3.0],[2.0,4.0]]" or "[[2.1],[3.5],[1.7],[4.0]]"''')
    
    args = parser.parse_args()

    model_path = os.path.join('./model', args.model_weight)
    params = extract_model_params(model_path)
    nprops = params["n_props"]

    prop_condition = args.prop_targets  # list of list
    # 检查维度是否匹配
    if len(prop_condition) > 0:
        if len(prop_condition[0]) != nprops:
            raise Exception(f"Error: The dimension of prop_targets ({len(prop_condition[0])}) != n_props ({nprops})")
    
    # 打印要生成的目标
    for i, cond in enumerate(prop_condition):
        print(f"Target {i+1}: {cond}")

    # 读取词表
    char_list = sorted(set(read_vocab(fname="./Voc_prior")) | set(['<','>']))
    stoi = {ch: i for i, ch in enumerate(char_list)}
    itos = {i: ch for i, ch in enumerate(char_list)}

    # 构建模型
    mconf = GPTConfig(
        vocab_size=params["vocab_size"], 
        block_size=params["block_size"], 
        num_props=nprops,
        n_layer=params["n_layer"], 
        n_head=args.n_head, 
        n_embd=params["n_embd"],
        lstm=args.lstm, 
        lstm_layers=args.lstm_layers
    )
    model = GPT(mconf)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading model...')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print('Model loaded.')

    gen_iter = math.ceil(args.gen_size / args.batch_size)
    all_slices = []

    # 对每组目标进行生成
    with torch.no_grad():
        for cond in prop_condition:
            print(f"\nGenerating for prop condition = {cond} ...")
            
            for _ in tqdm(range(gen_iter)):
                # 准备上下文
                context = '>'
                x = torch.tensor([stoi[context]], dtype=torch.long).unsqueeze(0).repeat(args.batch_size, 1).to(device)

                # 准备属性
                cond_tensor = torch.tensor(cond, dtype=torch.float, device=device)
                # shape = [batch_size, 1, nprops] 或 [batch_size, nprops]，模型 forward 里会处理
                if nprops == 1:
                    # [batch_size, 1]
                    p = cond_tensor.unsqueeze(1).repeat(args.batch_size, 1)
                else:
                    # [batch_size, nprops]
                    p = cond_tensor.unsqueeze(0).repeat(args.batch_size, 1)
                # 也可写成 p.unsqueeze(1) if GPT 里是 [B,1,nprops] 需求

                # 采样
                y = model.sample(
                    x, 
                    steps=params["block_size"], 
                    temperature=0.9, 
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.9, 
                    prop=p
                )
                
                # 解析生成的 token
                for gen_slice in y:
                    completion = " ".join([itos[int(tok)] for tok in gen_slice])
                    completion = completion.replace("<","").strip()
                    
                    # 检查有效性
                    if check_SLICES(completion, 4):
                        # e.g. 假设我们前两个性质是 [cond[0], cond[1]], 这里自行决定列名
                        # 也可以把 cond 转成 "prop1=xxx, prop2=yyy" 这种形式
                        all_slices.append({
                            # 示例：只打印
                            f"prop_{i}": cond[i] for i in range(len(cond))
                            # 但是上面这样写要注意 python 3.8+ 以上的语法
                        } | {"SLICES": completion})
                        
                        # 或者逐一写:
                        # item = {"SLICES": completion}
                        # for j in range(len(cond)):
                        #     item[f"prop_{j+1}"] = cond[j]
                        # all_slices.append(item)
                    else:
                        # 无效 SLICES
                        pass
    
    results = pd.DataFrame(all_slices)
    print(f"Total generated SLICES: {len(results)}")

    # Canonical & deduplicate
    if not results.empty:
        results['canonical_SLICES'] = results['SLICES'].apply(lambda x: get_canonical_SLICES(x, 4))
        results.drop_duplicates(subset='canonical_SLICES', inplace=True)
        results.drop('canonical_SLICES', axis=1, inplace=True)
        print(f"Unique canonical SLICES: {len(results)}")

    # 保存
    results.to_csv(f'{args.output_csv}', index=False)
    print(f"Saved CSV: {args.output_csv}")

    # 最后输出 ratio
    total_tries = args.batch_size * gen_iter * len(prop_condition)
    valid_ratio = len(all_slices) / total_tries if total_tries else 0
    unique_ratio = len(results) / len(all_slices) if len(all_slices) else 0
    print(f"Valid ratio: {valid_ratio:.3f}")
    print(f"Unique ratio: {unique_ratio:.3f}")
