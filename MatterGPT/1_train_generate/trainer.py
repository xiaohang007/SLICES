"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
with updated LR scheduling: if a validation dataset exists, use ReduceLROnPlateau (with patience for early stopping),
otherwise use CosineAnnealingLR.
"""

import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from utils import *
import re
import pandas as pd


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights

    # learning rate decay params: 如果 lr_decay 为 True，则采用调度器机制
    lr_decay = True
    # 以下参数用于新的调度器机制（不是基于 token 的调度）
    warmup_epochs = 2
    warmup_start_lr = 1e-5
    final_lr = 1e-5   # CosineAnnealing 中的最低 lr，或 ReduceLROnPlateau 的 min_lr
    patience = 5      # 如果有验证集，达到 patience 次数后触发早停

    # 旧版 token-based 参数（如果不使用 lr_decay 可忽略）
    warmup_tokens = 375e6
    final_tokens = 260e9

    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.stoi = stoi
        self.itos = itos

        # 设备设置
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        # 初始化优化器（注意：这里假设模型中实现了 configure_optimizers 方法）
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(config)

        # 如果 lr_decay 为 True，则使用调度器调度学习率
        if config.lr_decay:
            # 初始学习率设为 warmup_start_lr
            initial_lr = getattr(config, "warmup_start_lr", config.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = initial_lr

            # 如果存在验证集，使用 ReduceLROnPlateau 调度器（带 patience 机制）
            if self.val_dataset is not None:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.1,
                    patience=getattr(config, "patience", 1),
                    verbose=True,
                    min_lr=getattr(config, "final_lr", 1e-5)
                )
            else:
                # 没有验证集，则使用 CosineAnnealingLR
                warmup_epochs = getattr(config, "warmup_epochs", 0)
                T_max = max(1, config.max_epochs - warmup_epochs)
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=getattr(config, "final_lr", 1e-5)
                )
        else:
            self.scheduler = None

        # 早停相关（仅在有验证集时使用）
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 混合精度训练的 scaler
        self.scaler = GradScaler()


    def save_checkpoint(self):
        # DataParallel 封装下真实模型在 module 属性中
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)


    def run_epoch(self, split, epoch):
        is_train = split == 'train'
        self.model.train(is_train)
        dataset = self.train_dataset if is_train else self.val_dataset
        loader = DataLoader(dataset, shuffle=is_train, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

        for it, (x, y, p, s) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            p = p.to(self.device)
            s = s.to(self.device)
            
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(is_train):
                    logits, loss, _, _ = self.model(x, y, p, s)
            
            loss = loss.mean()  # collapse losses from multiple GPUs if necessary
            losses.append(loss.item())

            if is_train:
                self.model.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                pbar.set_description(f"Epoch {epoch+1} Iter {it}: Train Loss {loss.item():.5f}")

        avg_loss = float(np.mean(losses))
        print(f"Epoch {epoch+1} {split.capitalize()} - Loss: {avg_loss:.4f}")
        return avg_loss


    def train(self):
        config = self.config
        for epoch in range(config.max_epochs):
            train_loss = self.run_epoch('train', epoch)
            if self.val_dataset is not None:
                val_loss = self.run_epoch('val', epoch)
            else:
                val_loss = float('inf')

            # 学习率更新
            if config.lr_decay and self.scheduler is not None:
                if epoch < getattr(config, "warmup_epochs", 0):
                    # Warmup 阶段：线性增加学习率
                    warmup_lr = config.warmup_start_lr + (config.learning_rate - config.warmup_start_lr) * (epoch + 1) / config.warmup_epochs
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                    print(f"Warmup Phase: Setting learning rate to {warmup_lr:.7f}")
                else:
                    # 如果存在验证集，则 ReduceLROnPlateau 根据验证集 loss 调整 lr；否则 CosineAnnealing 直接 step()
                    if self.val_dataset is not None:
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # 提前停止：仅在有验证集时触发
            if self.val_dataset is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if config.ckpt_path is not None:
                        print(f"Saving at epoch {epoch+1} with best val loss {val_loss:.5f}")
                        self.save_checkpoint()
                else:
                    self.patience_counter += 1
                    print(f"No improvement in Val Loss. Patience: {self.patience_counter}/{config.patience}")
                if self.patience_counter >= config.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                if config.ckpt_path is not None:
                    self.save_checkpoint()
                    print(f"Model saved at epoch {epoch + 1}")

        return None

