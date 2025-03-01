# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
"""
Trainer for MatterGPT with xVal integration.
"""

import math
import os
import csv
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 导入调度器
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt

class TrainerConfig:
    # Optimization parameters
    max_epochs = 35  # 根据你的训练日志，调整为实际需要的 epoch 数量
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # Only applied on matmul weights
    # Learning rate decay params: linear warmup followed by ReduceLROnPlateau
    lr_decay = True  # 启用学习率衰减
    warmup_epochs = 2  # 指定 Warmup 的 epoch 数量
    warmup_start_lr = 1e-5  # Warmup 阶段的起始学习率
    final_lr = 1e-5  # ReduceLROnPlateau 的最小学习率
    # Checkpoint settings
    ckpt_path = "best_model.pth"  # 确保指定一个保存路径
    num_workers = 0  # For DataLoader
    patience = 5  # Early Stopping 的 patience

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config, stoi, itos, num_props):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Device configuration
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos
        self.num_props = num_props
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 删除现有的 train.log 文件
        if os.path.exists('train.log'):
            os.remove('train.log')

        # 打开 train.log 文件
        self.csvfile = open('train.log', 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
        self.writer.writeheader()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        # 初始化优化器和调度器
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(config)

        # 设置优化器的初始学习率为 warmup_start_lr
        if config.lr_decay:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = config.warmup_start_lr

            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1, 
                patience=1,  # 可根据需要调整
                verbose=True,
                min_lr=config.final_lr
            )
        else:
            self.scheduler = None

        self.scaler = GradScaler()

    def log_metrics(self, epoch, train_loss, val_loss, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        # 写入日志
        self.writer.writerow({
            'Epoch': epoch + 1,
            'Train Loss': f"{train_loss:.4f}",
            'Val Loss': f"{val_loss:.4f}",
            'Learning Rate': f"{lr:.7f}"
        })
        self.csvfile.flush()  # 确保数据被写入文件

    def save_checkpoint(self):
        # 保存模型权重
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model

        def run_epoch(split, epoch):
            is_train = split == 'train'  # split == 'train' 表示训练
            model.train(is_train)
            data = self.train_dataset if is_train else self.val_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            total_loss = []

            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                if self.num_props > 0:
                    # 假设 batch = (x, y, p)
                    x, y, p = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    p = p.to(self.device)
                    
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            # 调整为模型的 forward 方法
                            logits, loss, _, _ = model(x, y, p)
                else:
                    # 假设 batch = (x, y)
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss, _, _ = model(x, y)

                loss = loss.mean()
                total_loss.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # 更新进度条描述
                    pbar.set_description(f"Epoch {epoch+1} Iter {it}: Train Loss {loss.item():.5f}")

            # 计算平均损失
            avg_loss = float(np.mean(total_loss))

            if is_train:
                phase = "Train"
            else:
                phase = "Val"
                
            # 打印 epoch 总结
            print(f"Epoch {epoch+1} {phase} - Loss: {avg_loss:.4f}")

            return avg_loss

        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train', epoch)
            if self.val_dataset is not None:
                val_loss = run_epoch('val', epoch)
                print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Warmup 阶段：线性增加学习率
                if epoch < config.warmup_epochs and config.lr_decay:
                    warmup_lr = config.warmup_start_lr + (config.learning_rate - config.warmup_start_lr) * (epoch + 1) / config.warmup_epochs
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                    print(f"Warmup Phase: Setting learning rate to {warmup_lr:.7f}")
                elif config.lr_decay:
                    # Warmup 结束后，使用 ReduceLROnPlateau 调度器
                    self.scheduler.step(val_loss)

                # Early Stopping 逻辑
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint()
                    print(f"New best model saved with Val Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"No improvement in Val Loss. Patience: {self.patience_counter}/{config.patience}")

                if self.patience_counter >= config.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                val_loss = float('inf')
                print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
                # 在没有 val_dataset 时，每个 epoch 保存一次模型
                if config.ckpt_path is not None:
                    self.save_checkpoint()
                    print(f"Model saved at epoch {epoch + 1}")

            # 获取当前学习率
            lr = self.optimizer.param_groups[0]['lr']
            self.log_metrics(epoch, train_loss, val_loss, lr)

            # 可选的绘图代码：双Y轴显示损失和学习率
            fig, ax1 = plt.subplots(figsize=(10, 5))

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.plot(range(1, len(self.train_losses)+1), self.train_losses, label='Train Loss', color='tab:blue')
            if self.val_losses:
                ax1.plot(range(1, len(self.val_losses)+1), self.val_losses, label='Val Loss', color='tab:orange')
            ax1.tick_params(axis='y')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
            ax2.plot(range(1, len(self.learning_rates)+1), self.learning_rates, label='Learning Rate', color='tab:green')
            ax2.tick_params(axis='y')
            ax2.legend(loc='upper right')

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig('loss_curves.png')
            plt.close()

        # 训练完成后关闭 train.log 文件
        self.csvfile.close()
        return None
