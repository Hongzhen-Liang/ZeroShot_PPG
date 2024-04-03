import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size
        # 假设特征向量维度为 128
        self.img_features = torch.randn(size, 128)
        self.txt_features = torch.randn(size, 128)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.img_features[idx], self.txt_features[idx]

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        # 线性层将图像和文本特征映射到共同的嵌入空间
        self.img_linear = nn.Linear(feature_dim, embed_dim)
        self.txt_linear = nn.Linear(feature_dim, embed_dim)
    
    def forward(self, img_feature, txt_feature):
        img_embed = self.img_linear(img_feature)
        txt_embed = self.txt_linear(txt_feature)
        return img_embed, txt_embed

def contrastive_loss(img_embed, txt_embed, margin=1.0):
    # 使用余弦相似度计算匹配和不匹配对之间的距离
    cos = nn.CosineSimilarity(dim=1)
    positive_similarity = cos(img_embed, txt_embed)
    # 对于简化的例子，我们假设每个批次的第一个样本是正样本对，其余为负样本对
    negative_similarity = cos(img_embed[0].unsqueeze(0), txt_embed[1:])
    negative_similarity = negative_similarity.clamp(min=0.0)
    # 对比损失
    loss = 1 - positive_similarity + negative_similarity.mean()
    return loss.clamp(min=0.0)

# 参数设置
feature_dim = 128
embed_dim = 64
batch_size = 32
epochs = 10

# 数据加载
dataset = SimpleDataset(100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型初始化
model = ContrastiveModel(feature_dim, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(epochs):
    for img_features, txt_features in dataloader:
        optimizer.zero_grad()
        img_embed, txt_embed = model(img_features, txt_features)
        loss = contrastive_loss(img_embed, txt_embed)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
