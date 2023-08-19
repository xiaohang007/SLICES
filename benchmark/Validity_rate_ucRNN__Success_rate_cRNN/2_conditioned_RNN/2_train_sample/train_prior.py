#!usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
from torch.nn.parallel import DataParallel
import gc,re,subprocess
gc.collect()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size_now=128 # 38*8

def pretrain(restore_from=None):
    "Train the Prior RNN"

    # Reads vocabulary from a file
    voc = Vocabulary(init_from_file="Voc_prior")

    # Create a Dataset from a SMILES file
    moldata = MolData("../1_augmentation/prior_aug_50x.sci", voc)
    data = DataLoader(moldata, batch_size=batch_size_now, shuffle=True, drop_last=True,
                     collate_fn=MolData.collate_fn)
    #print(moldata.smiles)

    Prior = RNN(voc)
    #Prior = DataParallel(Prior, device_ids=[0,1,2])
    print("OK")
    # Can restore from a  saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.loag(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)

    for epoch in range(1, 10):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, (smiles, energies) in tqdm(enumerate(data), total=len(data)):

            # Sample from Dataloader
            seqs = smiles.long()
            energies = energies.float()
            # Calculate loss
            log_p, _ = Prior.likelihood(seqs, energies)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 440 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write('*'*50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))
                #seqs, likelihood, _ = Prior.sample(batch_size_now)        #这句的问题
                torch.save(Prior.rnn.state_dict(), 'Prior_local.ckpt')
        # Save the prior
        torch.save(Prior.rnn.state_dict(), 'Prior_local_'+str(epoch)+'.ckpt')


if __name__ == '__main__':
    pretrain()



