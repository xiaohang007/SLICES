# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import os,csv
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from utils import *




class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos, num_props, train_prop_mean):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos
        self.num_props = num_props
        self.train_prop_mean=train_prop_mean
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        # Delete existing train.log file if it exists
        if os.path.exists('train.log'):
            os.remove('train.log')
        
        # Open train.log file in write mode
        self.csvfile = open('train.log', 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=['Epoch', 'Train Loss', 'Test Loss', 'Learning Rate'])
        self.writer.writeheader()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

    def log_metrics(self, epoch, train_loss, test_loss, lr):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.learning_rates.append(lr)
        # Write metrics to train.log file
        self.writer.writerow({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Test Loss': test_loss,
            'Learning Rate': lr
        })
        self.csvfile.flush()  # Flush the buffer to ensure data is written to the file

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train' # split == 'train' represents training
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                if self.num_props > 0:
                    x, y, p = batch
                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    p = p.to(self.device)
                    # forward the model
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss, _, _ = model(x, y, p)
                            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                            losses.append(loss.item())
                else:
                    x, y = batch
                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # forward the model
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss, _, _ = model(x, y)
                            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                            losses.append(loss.item())                    

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay


        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                print("test loss:",test_loss,'\n')
            else:
                test_loss = float('inf')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            if self.config.ckpt_path is not None and self.test_dataset is None:
                print(f'Saving at epoch {epoch + 1} with best train loss: {train_loss}')
                self.save_checkpoint()
            if self.config.ckpt_path is not None and self.test_dataset is not None and test_loss < best_loss:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1} with best test loss: {test_loss}')
                self.save_checkpoint()                
            
            lr = optimizer.param_groups[0]['lr']
            self.log_metrics(epoch, train_loss, test_loss, lr)
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
            plt.plot(range(len(self.test_losses)), self.test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss_curves.png')
            plt.close()
            with open('train.log', 'w', newline='') as csvfile:
                fieldnames = ['Epoch', 'Train Loss', 'Test Loss', 'Learning Rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for epoch in range(len(self.train_losses)):
                    writer.writerow({
                        'Epoch': epoch + 1,
                        'Train Loss': self.train_losses[epoch],
                        'Test Loss': self.test_losses[epoch],
                        'Learning Rate': self.learning_rates[epoch]
                    })
        # Close the train.log file after training is complete
        self.csvfile.close()
        return None



