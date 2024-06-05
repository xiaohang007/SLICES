#!usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import joblib
from data_structs import MolDataHead, Vocabulary
from model import RNNHead
from utils import Variable, decrease_learning_rate
from torch.nn.parallel import DataParallel
import gc,re,subprocess
import argparse
from slices import check_SLICES
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv
gc.collect()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def downstream(voc_dir,train_data_dir,valid_data_dir,prior_dir,downstream_dir,batch_size=128,epochs=10,learning_rate=0.002,scaler_type=0):
    "Train the downstream RNN"
    log_file = downstream_dir.split(".")[0]+'_training_log.csv'
    if os.path.exists(log_file):
        os.remove(log_file)
    with open(log_file, 'a', newline='') as csvfile:  
        fieldnames = ['epoch', 'step', 'train_loss', 'valid_loss', 'valid_rate','learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Reads vocabulary from a file
        voc = Vocabulary(init_from_file=voc_dir)

        # Create a Dataset from a SLICES file
        train_data = MolDataHead(train_data_dir, voc, scaler_type)
        train_loader = DataLoader(train_data, batch_size=round(batch_size*0.9), shuffle=True, drop_last=True,
                        collate_fn=MolDataHead.collate_fn)

        # 读取验证数据
        valid_data = MolDataHead(valid_data_dir, voc, scaler_type, load_scaler=True)
        valid_loader = DataLoader(valid_data, batch_size=round(batch_size*0.1), shuffle=False, drop_last=True,
                        collate_fn=MolDataHead.collate_fn)

        best_valid_loss = float('inf')
        downstream = RNNHead(voc)
        #Prior = DataParallel(Prior, device_ids=[0,1,2])
        print("OK")
        # Can restore from a  saved RNN
        if torch.cuda.is_available() and prior_dir:
            print("load prior weights")
            state_dict = torch.load(prior_dir)  # 加载预训练权重
            model_dict = downstream.rnn.state_dict()
            

            # 筛选出尺寸匹配的权重
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)  # 更新当前模型的权重
            downstream.rnn.load_state_dict(model_dict)  # 加载更新后的状态字典
        elif prior_dir:
            print("load prior weights")
            downstream.rnn.load_state_dict(torch.load(prior_dir, \
            map_location=lambda storage, loc: storage),strict=False)
        params = [
            {'params': downstream.rnn.embedding.parameters(), 'lr': learning_rate*0.1},
            {'params': downstream.rnn.dense.parameters(), 'lr': learning_rate*10},
            {'params': downstream.rnn.gru_1.parameters(), 'lr': learning_rate},
            {'params': downstream.rnn.gru_2.parameters(), 'lr': learning_rate*0.1},
            {'params': downstream.rnn.gru_3.parameters(), 'lr': learning_rate*0.01},
            {'params': downstream.rnn.linear.parameters(), 'lr': learning_rate}
        ]
        optimizer = torch.optim.Adam(params)

        for epoch in range(0, epochs):
            # When training on a few million compounds, this model converges
            # in a few of epochs or even faster. If model sized is increased
            # its probably a good idea to check loss against an external set of
            # validation SLICES to make sure we dont overfit too much.
            for step, (slices, energies) in tqdm(enumerate(train_loader), total=len(train_loader)):

                # Sample from Dataloader
                seqs = slices.long()
                energies = energies.float()
                mean_energy = energies.mean()           
                # Calculate loss
                log_p, _ = downstream.likelihood(seqs, energies)
                loss = - log_p.mean()

                # Calculate gradients and take a step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Every 500 steps we decrease learning rate and print some information
                if step % round(len(train_loader)/2) == 0 and step != 0:
                    decrease_learning_rate(optimizer, decrease_by=0.03)
                    with torch.no_grad():
                        tqdm.write('*'*50)
                        tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))
                        seqs, likelihood, _ = downstream.sample(64,mean_energy)
                        valid = 0
                        for i, seq in enumerate(seqs.cpu().numpy()):
                            slices = voc.decode(seq)
                            if check_SLICES(slices,strategy=4,dupli_check=True,graph_rank_check=True):
                                valid += 1
                            #if i < 5:
                                #tqdm.write(slices)
                        valid_rate=100 * valid / len(seqs)
                        tqdm.write("\n{:>4.1f}% valid SLICES".format(valid_rate))
                        del seqs, likelihood, _
                        tqdm.write('*'*50 + '\n')


            with torch.no_grad():
                total_valid_loss = 0
                for valid_step, (valid_slices, valid_energies) in enumerate(valid_loader):
                    valid_seqs = valid_slices.long()
                    valid_energies = valid_energies.float() 
                    valid_log_p, _ = downstream.likelihood(valid_seqs, valid_energies)
                    valid_loss = -valid_log_p.mean()
                    total_valid_loss += valid_loss.item()
                del valid_log_p, _

                avg_valid_loss = total_valid_loss / (valid_step+1)
                current_lr = optimizer.param_groups[0]['lr']
                writer.writerow({'epoch': epoch, 'step': step, 'train_loss': loss.item(),
                                 'valid_loss': avg_valid_loss, 'valid_rate': valid_rate,
                                 'learning_rate': current_lr})
                csvfile.flush()
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Valid Loss: {avg_valid_loss:.4f}, "
                      f"Valid Rate: {valid_rate:.2f}%, Learning Rate: {current_lr:.6f}")           
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    print("best_valid_loss",best_valid_loss,"saving model")
                    torch.save(downstream.rnn.state_dict(), downstream_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain for SLICES generation")
    parser.add_argument('--voc', action='store',
                        default='data/Voc_withda', help='Directory for the vocabulary')
    parser.add_argument('--train_data_dir', action='store', default='cano_acceptors_sli.csv',
                        help='Directory of the SLICES/properties file for downstream learning')
    parser.add_argument('--valid_data_dir', action='store', default='cano_acceptors_sli.csv',
                            help='Directory of the SLICES/properties file for downstream learning')
    parser.add_argument('--prior_model', action='store', default=None,
                        help='Directory of the prior trained RNN')
    parser.add_argument('--downstream_model',action='store', default='data/tf_model_acceptor_smi_tuneall2.ckpt',
                        help='Directory of the downstream model')
    parser.add_argument('--batch_size', action='store', default='128',type=float)
    parser.add_argument('--learning_rate', action='store', default='0.002',type=float)
    parser.add_argument('--epochs', action='store', default='10',type=int)
    parser.add_argument("--scaler", required=False, default=0, type=int, help="Scaler to use for regression. 0 for no scaling, 1 for min-max scaling, 2 for standard scaling. Default: 0")
    args = parser.parse_args()

    downstream(args.voc,args.train_data_dir,args.valid_data_dir,args.prior_model,args.downstream_model,args.batch_size,args.epochs,args.learning_rate,args.scaler)



