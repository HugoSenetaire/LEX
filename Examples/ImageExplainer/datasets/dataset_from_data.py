import torch
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl

np.random.seed(0)
torch.manual_seed(0)


      


class DatasetFromData(Dataset):
    def __init__(self, data, target, transforms = None, target_transforms = None, give_index = True, noise_function = None) -> None:
        self.data = data
        self.targets = target
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.noise_function = noise_function
        self.give_index = give_index
        self.optimal_S_train = None
        self.optimal_S_test = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = img.numpy()

        if self.noise_function is not None :
            img = self.noise_function(img)

        if self.transforms is not None:
            img = self.transforms(img)

        
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.give_index :
            return img, target, idx
        else :
            return img, target
      
