import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('downstream_local_training_log.csv')

# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 在第一个子图中绘制训练损失和验证损失
ax1.plot(data['epoch'], data['train_loss'], label='Training Loss')
ax1.plot(data['epoch'], data['valid_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

# 在第二个子图中绘制验证准确率
ax2.plot(data['epoch'], data['valid_rate'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Rate')
ax2.set_title('Validation Rate')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
