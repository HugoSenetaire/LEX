
import torch
import pickle as pkl
import os
from torch.utils.data import Dataset

class reconstructionDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.len = len(dataset)

    def __getitem__(self, index):
        """
        Change getitem to return the image twice
        """
        aux = self.dataset.__getitem__(index)
        output = (aux[0], aux[0])
        return output
    def __len__(self):
        return self.len

class AutoEncoderDataset():
    def __init__(self, dataset, **kwargs):
        self.original_dataset = dataset
        self.dataset_train = reconstructionDataset(dataset.dataset_train)
        self.dataset_test = reconstructionDataset(dataset.dataset_test)


    def get_dim_input(self,):
        return self.original_dataset.get_dim_input()

    def get_dim_output(self,):
        return self.original_dataset.get_dim_input()

