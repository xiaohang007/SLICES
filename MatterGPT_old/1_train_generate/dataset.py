import torch
from torch.utils.data import Dataset
import numpy as np
import re
import math

class SliceDataset(Dataset):
    """Custom PyTorch Dataset that takes a file containing Slices.
    Args:
        args: all the args
        data: the list of slices
        char_list: vocabulary of all unique characters in slices
        block_size: maximum length of slices by characters
        prop_list: list of properties
        sym_list: list of crystal system symbols (e.g., 'sym0', 'sym1', etc.)
    """
    def __init__(self, args, data, char_list, block_size, prop_list=None, sym_list=None):
        chars = sorted(list(set(char_list)))
        data_size, vocab_size = len(data), len(chars)
        
        print('data has %d slices, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop_list = prop_list
        
        # Crystal system mapping
        self.crystal_systems = {
            'triclinic': 0,    # 三斜晶系
            'monoclinic': 1,   # 单斜晶系
            'orthorhombic': 2, # 正交晶系
            'tetragonal': 3,   # 四方晶系
            'trigonal': 4,     # 三方晶系
            'hexagonal': 5,    # 六方晶系
            'cubic': 6         # 立方晶系
        }
        self.crystal_systems_reverse = {v: k for k, v in self.crystal_systems.items()}
        self.num_crystal_systems = len(self.crystal_systems)
        self.sym_list = sym_list
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 处理基础序列数据
        slices = self.data[idx].strip().split(" ")
        slices += [str('<')] * (self.max_len - len(slices))
        slices = [str('>')] + slices  # add '>' as initial token
        dix = [self.stoi[s] for s in slices]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        # 处理属性数据
        prop = self.prop_list[idx]
        propt = torch.tensor(prop, dtype=torch.float)

        # 处理晶系数据
        sym_str = self.sym_list[idx]
        symt = self._encode_crystal_system(sym_str)
        
        return x, y, propt, symt


    def _encode_crystal_system(self, sym_str):
        """Convert crystal system string to one-hot encoding"""
        if sym_str not in self.crystal_systems:
            raise ValueError(f"Unknown crystal system: {sym_str}, expected one of {list(self.crystal_systems.keys())}")
            
        # One-hot encoding
        encoding = torch.zeros(self.num_crystal_systems)
        encoding[self.crystal_systems[sym_str]] = 1.0
        return encoding

    def decode_crystal_system(self, encoding):
        """Convert one-hot encoding back to crystal system string"""
        if isinstance(encoding, torch.Tensor):
            encoding = encoding.numpy()
        idx = np.argmax(encoding)
        
        if idx not in self.crystal_systems_reverse:
            raise ValueError(f"Invalid crystal system index: {idx}")
            
        return self.crystal_systems_reverse[idx]
      