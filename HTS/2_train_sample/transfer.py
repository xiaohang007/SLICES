#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
from rdkit.Chem import AllChem
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate, unique
rdBase.DisableLog('rdApp.error')


def cano_smi_file(fname, outfn):
    """
    canonicalize smile file
    """
    out = open(outfn, 'w')
    with open (fname) as f:
        for line in f:
            smi = line.rstrip()
            can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            out.write(can_smi + '\n')
    out.close()


def train_model():
    """Do transfer learning for generating SMILES"""
    voc = Vocabulary(init_from_file='data/Voc')
    cano_smi_file('refined_smii.csv', 'refined_smii_cano.csv')
    moldata = MolData('refined_smii_cano.csv', voc)
    # Monomers 67 and 180 were removed because of the unseen [C-] in voc
    # DAs containing [se] [SiH2] [n] removed: 38 molecules
    data = DataLoader(moldata, batch_size=64, shuffle=True, drop_last=False,
                      collate_fn=MolData.collate_fn)
    transfer_model = RNN(voc)

    if torch.cuda.is_available():
        transfer_model.rnn.load_state_dict(torch.load('data/Prior.ckpt'))
    else:
        transfer_model.rnn.load_state_dict(torch.load('data/Prior.ckpt', map_location=lambda storage, loc: storage))

    # for param in transfer_model.rnn.parameters():
    #     param.requires_grad = False
    optimizer = torch.optim.Adam(transfer_model.rnn.parameters(), lr=0.001)

    for epoch in range(1, 10):

        for step, batch in tqdm(enumerate(data), total=len(data)):
            seqs = batch.long()
            log_p, _ = transfer_model.likelihood(seqs)
            loss = -log_p.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write('*'*50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                seqs, likelihood, _ = transfer_model.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100*valid/len(seqs)))
                tqdm.write("*"*50 + '\n')
                torch.save(transfer_model.rnn.state_dict(), "data/transfer_model2.ckpt")

        torch.save(transfer_model.rnn.state_dict(), "data/transfer_modelw.ckpt")


def sample_smiles(nums, outfn, until=False):
    """Sample smiles using the transferred model"""
    voc = Vocabulary(init_from_file='data/voc')
    transfer_model = RNN(voc)
    output = open(outfn, 'w')

    if torch.cuda.is_available():
        transfer_model.rnn.load_state_dict(torch.load('data/tf_model_da_cluster_with_script2.ckpt'))
    else:
        transfer_model.rnn.load_state_dict(torch.load('data/tf_model_da_cluster_with_script2.ckpt',
                                                    map_location=lambda storage, loc:storage))

    for param in transfer_model.rnn.parameters():
        param.requires_grad = False

    if not until:

        seqs, likelihood, _ = transfer_model.sample(nums)
        valid = 0
        double_br = 0
        unique_idx = unique(seqs)
        seqs = seqs[unique_idx]
        for i, seq in enumerate(seqs.cpu().numpy()):

            smile = voc.decode(seq)
            if Chem.MolFromSmiles(smile):
                try:
                    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, 1024)
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
            seq = seq.cpu().numpy()
            seq = seq[0]
            # print(seq)
            smile = voc.decode(seq)
            if Chem.MolFromSmiles(smile):
                try:
                    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, 1024)
                    valid += 1
                    output.write(smile + '\n')
                    if valid % 100 == 0 and valid != 0:
                        tqdm.write('\n{} valid molecules sampled, with {} of total samples'.format(valid, n_sample))
                except:
                    continue
        tqdm.write('\n{} valid molecules sampled, with {} of total samples'.format(nums, n_sample))


if __name__ == "__main__":
    train_model()
    #for i in [1024, 2048, 4096, 8192, 16384, 32768]:
    for i in [1024]:
        sample_smiles(i, 'sampled_da_'+str(i)+'.csv')
    #sample_smiles(10000, 'sampled_da_cluster_untill_10000.csv', until=True)