import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd 

class Feeder(Dataset):
    def __init__(self, X, Y):
        '''
            X: (n, D, 2L, V); Y: (n,)
        '''
        self.num_cls = 2
        self.data = X
        self.labels = Y
        # self.subs = subs

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).permute(1,2,0) #(2L, V, D)
        L = x.shape[0]//2
        x = [x[:L,:,:], x[L:, :, :]]
        y = torch.from_numpy(self.labels[index])
        # curr_sub = self.subs[index]

        return x, y/100.#, curr_sub


