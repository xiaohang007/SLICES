#!usr/bin/env python

import torch,os
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
import gc,re,subprocess
import argparse
gc.collect()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def pretrain(restore_from=None,batch_size=128,epochs=10):
    "Train the Prior RNN"
    print("Train the Prior RNN")
    # Reads vocabulary from a file
    voc = Vocabulary(init_from_file="Voc_prior")

    # Create a Dataset from a SMILES file
    moldata = MolData("../1_augmentation/prior_aug.sli", voc)
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                     collate_fn=MolData.collate_fn)
    #print(moldata.smiles)

    Prior = RNN(voc)
    # Can restore from a  saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.loag(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)

    for epoch in range(0, epochs):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from Dataloader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 700 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write('*'*50)
                #tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                seqs, likelihood, _ = Prior.sample(batch_size)        #这句的问题
                torch.save(Prior.rnn.state_dict(), 'Prior_local.ckpt')
        # Save the prior
        torch.save(Prior.rnn.state_dict(), 'Prior_local.ckpt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain for SLICES generation")
    parser.add_argument('--batch_size', action='store', dest='nums', default='128')
    parser.add_argument('--epochs', action='store', dest='epochs', default='10')
    arg_dict = vars(parser.parse_args())
    nums_,epochs_= arg_dict.values()
    pretrain(restore_from=None,batch_size=int(nums_),epochs=int(epochs_))



