from utils import *
from dataset import SliceDataset
from model import MatterGPT, MatterGPTConfig
import math
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import json
import os
import sys
from slices import check_SLICES, get_canonical_SLICES
import configparser
import ast
from model import FlashCausalSelfAttention

def get_crystal_system_distribution(train_file):
    """计算训练集中晶系的分布"""
    df = pd.read_csv(train_file)
    distribution = df['crystal_system'].value_counts(normalize=True)
    return distribution


def sample_crystal_system(distribution):
    """根据分布随机采样晶系"""
    return np.random.choice(distribution.index, p=distribution.values)


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
        return [[float(x) for x in row] for row in parsed]
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid prop_condition format: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--output_csv', type=str, help="name to save the generated mols in csv format", required=True)
    parser.add_argument('--batch_size', type=int, default = 128, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
    parser.add_argument('--sym_dim', type=int, default=7,
                      help="dimension of symmetry embedding", required=False)
    parser.add_argument('--train_dataset', type=str, default="../0_dataset/train_data.csv", help="path to the training dataset file")
    parser.add_argument('--prop_targets', type=parse_prop_condition, required=True,
                      help='''Property condition as a string representation of a list of lists.
                      e.g., "[[1.0,3.0],[2.0,4.0]]" or "[[2.1],[3.5],[1.7],[4.0]]"''')

    args = parser.parse_args()
    
    # 已移除冗余的模型结构参数，模型结构信息将从保存的 ini 文件中加载

    # load model       
    model_name = os.path.splitext(args.model_weight)[0]
    config_path = os.path.join('./model', f'{model_name}.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    model_config = configparser.ConfigParser()
    model_config.read(config_path)
    mp = model_config['ModelParameters']

    # Load vocabulary
    vocab_path = os.path.join('./model', f'{model_name}_vocab.json')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        stoi = {k: int(v) for k, v in vocab_data['stoi'].items()}
        itos = {int(k): v for k, v in vocab_data['itos'].items()}
        
    # Create char_list from vocabulary
    char_list = list(stoi.keys())

    mconf = MatterGPTConfig(
        vocab_size = int(mp['vocab_size']),
        block_size = int(mp['block_size']),
        num_props = int(mp['num_props']),
        sym_dim = args.sym_dim,
        n_layer = int(mp['n_layer']),
        n_head = int(mp['n_head']),
        n_embd = int(mp['n_embd'])
    )
    
    
    
    model = MatterGPT(mconf)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading model')
    model.load_state_dict(torch.load('./model/' + args.model_weight, map_location=torch.device(device)))
    model.to(device)
    # Convert relevant layers' weights to bfloat16
    for name, module in model.named_modules():
        if isinstance(module, FlashCausalSelfAttention):
            for param in module.parameters():
                param.data = param.data.to(torch.bfloat16)
    model.eval()
    print('Model loaded.')

    # 创建数据集实例以获取晶系编码/解码功能
    dataset = SliceDataset(args, [], char_list,  int(mp['block_size']))


    # 获取训练集的晶系分布
    train_distribution = get_crystal_system_distribution(args.train_dataset)
    print("\nTraining set crystal system distribution:")
    for sys, prob in train_distribution.items():
        print(f"{sys}: {prob:.3f}")



    # Replace prop_conditions with args.prop_targets
    prop_conditions = args.prop_targets  # Now using the parsed property targets
    
    gen_iter = math.ceil(args.gen_size / args.batch_size)     
    all_results = []
    model.eval()
    with torch.inference_mode():
        for prop in prop_conditions:
            slices = []
            print(f"\nGenerating for prop condition = {prop}")
            
            for i in tqdm(range(gen_iter)):
                context = '>'
                x = torch.tensor([stoi[context]], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
                
                # 设置属性条件
                p = torch.tensor(prop).repeat(args.batch_size, 1).to(device)
                
                # 为每个batch中的样本随机采样晶系
                sym_types = [sample_crystal_system(train_distribution) for _ in range(args.batch_size)]
                s = torch.stack([dataset._encode_crystal_system(sym) for sym in sym_types]).to(device)
                
                # 生成结构
                y = model.sample(x, int(mp['block_size']), 
                        temperature=0.9, 
                        do_sample=True, 
                        top_k=50, 
                        top_p=0.9, 
                        prop=p,
                        sym=s)

                for gen_mol, sym_type in zip(y, sym_types):
                    completion = " ".join([itos[int(i.item())] for i in gen_mol])
                    completion = completion.replace("<", "").strip()
                    is_crystal = check_SLICES(completion,4)
                    if is_crystal:
                        slices.append({
                            # 示例：只打印
                            f"prop_{i}": prop[i] for i in range(len(prop))
                            # 但是上面这样写要注意 python 3.8+ 以上的语法
                        } | {'crystal_system': sym_type,"SLICES": completion})
                    else:
                        pass
            
            results_df = pd.DataFrame(slices)
            all_results.append(results_df)
            
            # 打印当前生成结果的晶系分布
            current_distribution = results_df['crystal_system'].value_counts(normalize=True)
            for sys, prob in current_distribution.items():
                target_prob = train_distribution.get(sys, 0)

            
            # 打印其他统计信息
            unique_slices = list(set(results_df['SLICES']))
            print('\nGeneration Statistics:')
            print('Valid ratio: ', np.round(len(results_df)/(args.batch_size*gen_iter), 3))
            print('Unique ratio: ', np.round(len(unique_slices)/len(results_df), 3))

    # 合并所有结果并保存
    final_results = pd.concat(all_results, ignore_index=True)
    
    # 比较最终生成结果与训练集的分布差异
    final_distribution = final_results['crystal_system'].value_counts(normalize=True)
    for sys in set(train_distribution.index) | set(final_distribution.index):
        train_prob = train_distribution.get(sys, 0)
        gen_prob = final_distribution.get(sys, 0)
    
    final_results.to_csv(args.output_csv, index=False)

    # 打印总体统计信息
    unique_slices = list(set(final_results['SLICES']))
    total_attempts = args.batch_size * gen_iter * len(prop_conditions)

    total_attempts = args.batch_size * gen_iter * len(prop_conditions)
    print('\nOverall Statistics:')
    print('Total valid ratio: ', np.round(len(final_results)/total_attempts, 3))

