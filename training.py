import os 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import cv2 as cv
from model.trans import CNNFormer
from utils.data import TrajectoriesData

def ambil_data(folder):
    gecko = []
    dir_list = os.listdir(folder)
    for i in dir_list :
      data = cv.imread(folder + '/' + i)
      data = cv.resize(data,(300,300))
      data = data/255
      gecko.append(data)
    return gecko

data_a = np.array(ambil_data('D:/SEKOLAH/KULIAH/SEMESTER 6/PENGENALAN POLA/21102254_Karsten Errando Winoto_IF 09 SC 1_Pengenalan Pola 1/DS_GEKCO/ALBINO'))
data_b = np.array(ambil_data('D:/SEKOLAH/KULIAH/SEMESTER 6/PENGENALAN POLA/21102254_Karsten Errando Winoto_IF 09 SC 1_Pengenalan Pola 1/DS_GEKCO/BOLDSTIPE_ALBINO'))
data_ab = np.array(ambil_data('D:/SEKOLAH/KULIAH/SEMESTER 6/PENGENALAN POLA/21102254_Karsten Errando Winoto_IF 09 SC 1_Pengenalan Pola 1/DS_GEKCO/BOLDSTRIPE'))

train_data = torch.utils.data.DataLoader(TrajectoriesData([
    (data_a[:16],0),
    (data_b[:16],1),
    (data_ab[:16],2)
    ]),batch_size=8,shuffle = True)
test_data = torch.utils.data.DataLoader(TrajectoriesData([
    (data_a[16:],0),
    (data_b[16:],1),
    (data_ab[16:],2)]),batch_size=8,shuffle = True)

model = CNNFormer(feature_dim = 3 )
model.to(device='cpu')
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion  = nn.CrossEntropyLoss()
EPOCH = 100
device = 'cpu'
loss_all = []
total_correct = 0
total_samples = 0
for epoch in range(EPOCH):
    loss_total = 0
    for batch, (src, trg) in enumerate(train_data):
        #print(src.shape)
        #print(trg.shape)
        src = src.permute(0,3,1,2)
        pred = model(src).to(device)
        loss = criterion(pred, trg.to(device))
        loss_total+=loss.item()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_correct += (torch.argmax(pred, dim=1) == torch.argmax(trg, dim=1)).sum().item()
        total_samples += trg.size(0)
    loss_batch = loss_total / len(train_data)
    loss_all.append(loss_batch)
    accuracy = total_correct / total_samples
    print(f'Epoch {epoch+1}/{EPOCH}, Loss: {loss_batch:.4f}, Accuracy: {accuracy:.4f}')