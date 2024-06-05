#!usr/bin/env python

import torch,os
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import argparse
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
import gc,re,subprocess
from slices import check_SLICES
import csv
gc.collect()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def pretrain(voc_dir,train_data_dir,valid_data_dir,prior_dir,batch_size=128,epochs=10,restore_from=None):
    "Train the Prior RNN"
    log_file = 'prior_training_log.csv'
    if os.path.exists(log_file):
        os.remove(log_file)
    with open('prior_training_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'step', 'train_loss', 'valid_loss', 'valid_rate','learning_rate','best_valid_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Reads vocabulary from a file
        voc = Vocabulary(init_from_file=voc_dir)

        # Create a Dataset from a SLICES file
        train_data = MolData(train_data_dir, voc)
        train_loader = DataLoader(train_data, batch_size=round(batch_size*0.9), shuffle=True, drop_last=True,
                         collate_fn=MolData.collate_fn)

        valid_data = MolData(valid_data_dir, voc)
        valid_loader = DataLoader(valid_data, batch_size=round(batch_size*0.1), shuffle=False, drop_last=True,
                         collate_fn=MolData.collate_fn)
        
        #best_valid_loss = float('inf')
        #print(moldata.smiles)
        best_valid_rate=0

        Prior = RNN(voc)
        print("OK")
        # Can restore from a  saved RNN
        if restore_from:
            Prior.rnn.load_state_dict(torch.loag(restore_from))

        optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)

        for epoch in range(0, epochs):
            # When training on a few million compounds, this model converges
            # in a few of epochs or even faster. If model sized is increased
            # its probably a good idea to check loss against an external set of
            # validation SMILES to make sure we dont overfit too much.
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

                # Sample from Dataloader
                seqs = batch.long()

                # Calculate loss
                log_p, _ = Prior.likelihood(seqs)
                loss = - log_p.mean()

                # Calculate gradients and take a step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del log_p, seqs
                # Every 500 steps we decrease learning rate and print some information
                if step % round(len(train_loader)/5) == 0 and step != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.03)
                    with torch.no_grad():
                        tqdm.write('*'*50)
                        tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))
                        del loss
                        seqs, likelihood, _ = Prior.sample(64)
                        valid = 0
                        for i, seq in enumerate(seqs.cpu().numpy()):
                            slices = voc.decode(seq)
                            if check_SLICES(slices,strategy=4,dupli_check=True,graph_rank_check=True):
                                valid += 1
                            #if i < 5:
                                #tqdm.write(slices)
                        torch.cuda.empty_cache()
                        valid_rate=100 * valid / len(seqs)
                        del seqs, likelihood, _
                        tqdm.write("\n{:>4.1f}% valid SLICES".format(valid_rate))
                        if valid_rate >= best_valid_rate:
                            best_valid_rate = valid_rate
                            print("best_valid_rate",best_valid_rate,"saving model")
                            torch.save(Prior.rnn.state_dict(), prior_dir)

                        tqdm.write('*'*50 + '\n')


            with torch.no_grad():
                total_valid_loss = 0
                for valid_step, valid_batch in enumerate(valid_loader):
                    valid_seqs = valid_batch.long()
                    valid_log_p, _ = Prior.likelihood(valid_seqs)
                    valid_loss = -valid_log_p.mean()
                    total_valid_loss += valid_loss.item()
                del valid_log_p, _
                
                avg_valid_loss = total_valid_loss / (valid_step+1)
                current_lr = optimizer.param_groups[0]['lr']

                writer.writerow({'epoch': epoch, 'step': step, 'train_loss': loss.item(),
                                 'valid_loss': avg_valid_loss, "best_valid_rate": best_valid_rate,
                                 'learning_rate': current_lr})
                csvfile.flush()
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Valid Loss: {avg_valid_loss:.4f}, "
                      f"Learning Rate: {current_lr:.6f}","best_valid_rate",best_valid_rate)
                torch.save(Prior.rnn.state_dict(), 'final_'+prior_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain for SLICES generation")
    parser.add_argument('--voc', action='store',
                        default='data/Voc_withda', help='Directory for the vocabulary')
    parser.add_argument('--train_data_dir', action='store', default='cano_acceptors_sli.csv',
                        help='Directory of the SLICES/properties file for downstream learning')
    parser.add_argument('--valid_data_dir', action='store', default='cano_acceptors_sli.csv',
                        help='Directory of the SLICES/properties file for downstream learning')
    parser.add_argument('--prior_model', action='store', default='data/Prior_gua_withda.ckpt',
                        help='Directory of the prior trained RNN')
    parser.add_argument('--batch_size', action='store', default='128', type=float)
    parser.add_argument('--epochs', action='store', default='10', type=int)
    args = parser.parse_args()
    pretrain(args.voc,args.train_data_dir,args.valid_data_dir,args.prior_model,args.batch_size,args.epochs)

