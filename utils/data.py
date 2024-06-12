import os 
import torch 
import cv2 as cv
import numpy as np

from torch.utils.data import Dataset, DataLoader
class TrajectoriesData(torch.utils.data.Dataset):
    def __init__(self, data):
        to1hot = np.eye(3)
        self.dataset = []
        for d, label in data:
            #print(d, label)
            self.dataset += [
                (im, to1hot[label]) for im in d
            ]
        #print(len(self.dataset))
    def __len__(self):
            return len(self.dataset)

    def __getitem__(self, index:int) -> (np.ndarray, int):
            vec, label = self.dataset[index]
            return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)







