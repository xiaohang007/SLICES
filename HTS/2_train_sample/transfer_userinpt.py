#!/usr/bin/env python
import numpy
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

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()



def train_model(voc_dir, smi_dir, prior_dir, tf_dir,tf_process_dir,freeze=False):
    """
    Transfer learning on target molecules using the SMILES structures
    Args:
        voc_dir: location of the vocabulary
        smi_dir: location of the SMILES file used for transfer learning
        prior_dir: location of prior trained model to initialize transfer learning
        tf_dir: location to save the transfer learning model
        tf_process_dir: location to save the SMILES sampled while doing transfer learning
        freeze: Bool. If true, all parameters in the RNN will be frozen except for the last linear layer during
        transfer learning.

    Returns: None

    """
    voc = Vocabulary(init_from_file=voc_dir)
    #cano_smi_file('all_smi_refined.csv', 'all_smi_refined_cano.csv')
    moldata = MolData(smi_dir, voc)
    # Monomers 67 and 180 were removed because of the unseen [C-] in voc
    # DAs containing [C] removed: 43 molecules in 5356; Ge removed: 154 in 5356; [c] removed 4 in 5356
    # [S] 1 molecule in 5356
    data = DataLoader(moldata, batch_size=32, shuffle=True, drop_last=False,
                      collate_fn=MolData.collate_fn)
    transfer_model = RNN(voc)
    # if freeze=True, freeze all parameters except those in the linear layer
    if freeze:
        for param in transfer_model.rnn.parameters():
            param.requires_grad = False
        transfer_model.rnn.linear = nn.Linear(512, voc.vocab_size)
    if torch.cuda.is_available():
        transfer_model.rnn.load_state_dict(torch.load(prior_dir))
    else:
        transfer_model.rnn.load_state_dict(torch.load(prior_dir,
                                                      map_location=lambda storage, loc: storage))

    optimizer = torch.optim.Adam(transfer_model.rnn.parameters(), lr=0.0005)

    smi_lst = []; epoch_lst = []
    for epoch in range(1, 8):

        for step, batch in tqdm(enumerate(data), total=len(data)):
            seqs = batch.long()
            log_p, _ = transfer_model.likelihood(seqs)
            loss = -log_p.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 40 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                #tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                tqdm.write("*"*50 + '\n')
                torch.save(transfer_model.rnn.state_dict(), tf_dir)
        torch.save(transfer_model.rnn.state_dict(), tf_dir)




def sample_smiles(voc_dir, nums, outfn,tf_dir, until=False):
    """Sample smiles using the transferred model"""
    voc = Vocabulary(init_from_file=voc_dir)
    transfer_model = RNN(voc)
    output = open(outfn, 'w')

    if torch.cuda.is_available():
        transfer_model.rnn.load_state_dict(torch.load(tf_dir))
    else:
        transfer_model.rnn.load_state_dict(torch.load(tf_dir,
                                                    map_location=lambda storage, loc:storage))

    for param in transfer_model.rnn.parameters():
        param.requires_grad = False

    if not until:

        seqs, likelihood, _ = transfer_model.sample(int(nums))
        valid = 0
        double_br = 0
        unique_idx = unique(seqs)
        seqs = seqs[unique_idx]
        with open('temp','w+') as f:
            x_df = pd.DataFrame(seqs)
            x_df.to_csv('tmp.csv')
        for i, seq in enumerate(seqs.cpu().numpy()):

            smile = voc.decode(seq)
            if 1:
                try:
                    #AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, 1024)
                    valid += 1
                    output.write(smile+'\n')
                except:
                    continue
            #if smile.count('Br') == 2:
            #    double_br += 1
            #output.write(smile+'\n')
        tqdm.write('\n{} molecules sampled, {} valid SMILES, {} with double Br'.format(nums, valid, double_br))
        output.close()
    else:
        valid = 0;
        n_sample = 0
        while valid < nums:
            seq, likelihood, _ = transfer_model.sample(1)
            n_sample += 1
            #seq = seq.cpu().numpy()
            #seq = seq[0]
            # print(seq)
            smile = voc.decode(seq)
            if 0:
                try:
                    #AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, 1024)
                    valid += 1
                    output.write(smile + '\n')
                    #if valid % 100 == 0 and valid != 0:
                    #    tqdm.write('\n{} valid molecules sampled, with {} of total samples'.format(valid, n_sample))
                except:
                    continue
        tqdm.write('\n{} valid molecules sampled, with {} of total samples'.format(nums, n_sample))



if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Transfer learning for SMILES generation")
    parser.add_argument('--task', action='store', dest='task', choices=['train_model', 'sample_smiles'],
                        default='train_model',help='What task to perform')
    parser.add_argument('--voc', action='store', dest='voc_dir',
                        default='data/Voc_withda', help='Directory for the vocabulary')
    parser.add_argument('--smi', action='store', dest='smi_dir', default='cano_acceptors_smi.csv',
                        help='Directory of the SMILES file for tranfer learning')
    parser.add_argument('--prior_model', action='store', dest='prior_dir', default='data/Prior_gua_withda.ckpt',
                        help='Directory of the prior trained RNN')
    parser.add_argument('--tf_model',action='store', dest='tf_dir', default='data/tf_model_acceptor_smi_tuneall2.ckpt',
                        help='Directory of the transfer model')
    parser.add_argument('--nums', action='store', dest='nums', default='64',
                        help='Number of SMILES to sample for transfer learning')
    parser.add_argument('--save_smi',action='store',dest='save_dir',default='acceptor_1024_tuneall2.csv',
                        help='Directory to save the generated SMILES')
    parser.add_argument('--save_process_smi',action='store',dest='tf_process_dir',default='Model1_sample_process.csv',
                        help='Directory to save the generated SMILES')
    arg_dict = vars(parser.parse_args())
    print(arg_dict)
    task_, voc_, smi_, prior_, tf_, nums_, save_smi_, tf_process_dir_ = arg_dict.values()

    if task_ == 'train_model':
        train_model(voc_dir=voc_, smi_dir=smi_, prior_dir=prior_, tf_dir=tf_,
                    tf_process_dir=tf_process_dir_,freeze=False)
    if task_ == 'sample_smiles':
        sample_smiles(voc_, nums_,save_smi_,tf_, until=False)


