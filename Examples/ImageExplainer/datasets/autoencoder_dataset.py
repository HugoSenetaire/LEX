
import torch
import pickle as pkl
import os
from torch.utils.data import Dataset
import numpy as np
class reconstructionDataset(Dataset):
    def __init__(self, dataset, add_noise=False):
        super().__init__()
        self.dataset = dataset
        self.len = len(dataset)
        self.add_noise = add_noise

    def __getitem__(self, index):
        """
        Change getitem to return the image twice
        """
        aux = self.dataset.__getitem__(index)
        input = aux[0]
        if not isinstance(input, torch.Tensor):
            input = torch.from_numpy(input)
    
        if self.add_noise:
            new_input = input + torch.randn_like(input) * 0.01
        else :
            new_input = input
        output = (new_input, input)
        return output
    def __len__(self):
        return self.len

class AutoEncoderDataset():
    def __init__(self, dataset, add_noise = False, **kwargs):
        self.original_dataset = dataset
        self.add_noise = add_noise
        self.dataset_train = reconstructionDataset(dataset.dataset_train, add_noise = add_noise)
        self.dataset_test = reconstructionDataset(dataset.dataset_test, add_noise = add_noise)


    def get_dim_input(self,):
        return self.original_dataset.get_dim_input()

    def get_dim_output(self,):
        return self.original_dataset.get_dim_input()

