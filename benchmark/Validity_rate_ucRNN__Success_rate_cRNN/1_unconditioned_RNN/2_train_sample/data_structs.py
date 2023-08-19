import numpy as np
import random
import re
import pickle
import sys
import time
import math
import torch
from torch.utils.data import Dataset

from utils import Variable

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=666):
        self.special_tokens = ['EOS', 'GO']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]

        return smiles_matrix
        
    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        if chars:
            smiles = " ".join(chars)           # add space between meter and other tokens
        else:
            print('Null Chars!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            smiles = "".join(chars) 
        return smiles
    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        tokenized = smiles.strip().split(' ')
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r',encoding='utf-8-sig') as f:
            for line in f:
                self.smiles.append(line.split('\n')[0])

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        if encoded is not None:
            return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr



def tokenize(smiles):
    """Takes a SMILES and return a list of characters/tokens"""
    tokenized = smiles.split(' ')
    tokenized.append('EOS')
    return tokenized

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.strip()
            #mol = Chem.MolFromSmiles(smiles)
            if 1:
                smiles_list.append(smiles)
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    return smiles_list_valid

def filter_file_on_chars(smiles_fname, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(smiles_fname + "_filtered", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")

def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        char_list=smiles.split(' ')
        for char in char_list:
            add_chars.add(char)

    print("Number of characters: {}".format(len(add_chars)))
    with open('Voc_prior', 'w') as f:
        voc_text=""
        for char in add_chars:
            voc_text+=char + "\n"
        f.write(voc_text[:-1])
    return add_chars

def can_smi_file(fname):
    """

    Args:
        fname:

    Returns:

    """
    out = open(fname+'cano', 'w')
    with open (fname) as f:
        for line in f:
            smi = line.rstrip()
            can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            out.write(can_smi + '\n')
    out.close()


def batch_iter(data, batch_size=128, shuffle=True):
    batch_num = math.ceil(len(data)/batch_size)
    idx_arr = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idx_arr)
    for i in range(batch_num):
        indices= idx_arr[i*batch_size: (i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e:len(e), reverse=True)
        yield i, examples


def pad_seq(seqs):
    batch_size = len(seqs)
    seq_lengths = torch.LongTensor(list(map(len,seqs)))
    max_length = len(seqs[0])
    pad_seq = torch.zeros(batch_size, max_length,dtype=torch.long)
    for i, seq in enumerate(seqs):
        pad_seq[i, :len(seq)] = seq
    return seq_lengths,pad_seq

def mask_seq(seqs, seq_lens):
    mask = torch.zeros(seqs.size(0),seqs.size(1))
    for i, length in enumerate(seq_lens):
        mask[i, 0:length] = seqs[i, 0:length]
    return mask

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_file(smiles_file)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(smiles_list)
    #write_smiles_to_file(smiles_list, "prior_2.sci")

