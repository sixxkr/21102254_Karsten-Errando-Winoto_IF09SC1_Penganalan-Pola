import numpy as np 
import cv2 as cv
import os 
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNNFormer(nn.Module):
  def __init__(self, feature_dim, dff=1024, num_head=1,num_layer=1, n_class=3, dropout=0.1, device='cpu'):
    super(CNNFormer, self).__init__()
    self.layer = num_layer
    self.conv = nn.Sequential(
      nn.Conv2d(feature_dim, 20, 2),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(20, 20, 5),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(20, 20, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(20, 20, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Dropout(p=dropout),
    )

    # Hitung hidden_dim berdasarkan jumlah filter terbanyak
    self.hidden_dim = 20
    self.MHA = nn.MultiheadAttention(embed_dim=self.hidden_dim,num_heads=num_head, bias=False, dropout=dropout).to(device)
    self.feed_forward = nn.Sequential(
      nn.Linear(self.hidden_dim, dff),
      nn.ReLU(),
      nn.Linear(dff, self.hidden_dim)
    )
    self.norm = nn.LayerNorm(self.hidden_dim)
    # Sesuaikan dimensi lapisan linear
    self.lin_out = nn.Linear(self.hidden_dim * 256, n_class) #Ubah input size
  def forward(self, x):
    # Layer convolution
    x = self.conv(x)
    # Ubah dimensi tensor untuk sesuai dengan input multi-head  attention
    batch_size, channels, height, width = x.size()
    x = x.view(batch_size, channels, -1).permute(0, 2, 1)
    # Layer transformer encoder
    for i in range(self.layer):
      y, _ = self.MHA(x, x, x)
      x = x + self.norm(y)
      y = self.feed_forward(x)
      x = x + self.norm(y)
      # Kembalikan dimensi ke format semula
      x = x.permute(0, 2, 1).view(batch_size, channels, height,
      width)
      # Ubah dimensi tensor untuk sesuai dengan lapisan linear
      x = x.reshape(batch_size, -1)
      # Layer linear output
      x = self.lin_out(x)
    return x