# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
   
import numpy as np
import threading

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def construct_vocabulary(slices_list):
    """Returns all the characters present in a SLICES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, slices in enumerate(slices_list):
        char_list = slices.strip().split(' ')
        for char in char_list:
            add_chars.add(char)

    print("Number of characters: {}".format(len(add_chars)))
    with open('Voc_prior', 'w') as f:
        voc_text = ""
        for char in add_chars:
            voc_text += char + "\n"
        f.write(voc_text[:-1])
    return add_chars


def read_slices_from_file(fname):
    """Reads a Slices """
    with open(fname, 'r') as f:
        slices_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            slices = line.strip()
            if 1:
                slices_list.append(slices)
        print("{} SLICES retrieved".format(len(slices_list)))
        return slices_list


def read_bandgap_from_file(fname):
    """Reads a Slices """
    with open(fname, 'r') as f:
        bandgap_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            bandgap = line.strip()
            if 1:
                bandgap_list.append(bandgap)
        print("{} bandgap retrieved".format(len(bandgap_list)))
        return bandgap_list

def read_formationenergy_from_file(fname):
    """Reads a Slices """
    with open(fname, 'r') as f:
        formationenergy_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            formationenergy = line.strip()
            if 1:
                formationenergy_list.append(formationenergy)
        print("{} formation energy_list retrieved".format(len(formationenergy_list)))
        return formationenergy_list

def read_vocab(fname="Voc_prior"):
    """Reads a Slices """
    additional_chars = set()
    print(fname)
    with open(fname, 'r') as f:
        chars = f.read().split()
    for char in chars:
        additional_chars.add(char)
    char_list = list(additional_chars)
    char_list.sort()
    #vocab_size = len(char_list)
    #vocab = dict(zip(char_list, range(len(char_list))))
    #reversed_vocab = {v: k for k, v in vocab.items()}
    return char_list


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()   
    model.eval()

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def check_novelty(gen_smiles, train_smiles): # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio




        


