import torch
import torchvision 
from torch.utils.data import Dataset
import numpy as np 
import os
import pickle as pkl






# FASHION MNIST :

class FashionMNISTDataset(torchvision.datasets.FashionMNIST):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            give_index=True,
            **kwargs,):


        super().__init__(root, train, transform, target_transform, download)
        self.give_index = give_index
        self.optimal_S_train = None
        self.optimal_S_test = None
        self.noisy = noisy
        self.noise_function = noise_function

       
    def __str__(self):
        return "SimpleMnist"
        
    def __getitem__(self, idx):
        if not self.noisy :
            img, target = self.data[idx], int(self.targets[idx])

            img = img.numpy()
            # target = target.numpy()
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            

        else :
            img, target = self.data[idx], self.data[idx]
            
            img = img.numpy()
            target = target.numpy()
      
            if self.transform is not None:
                target = self.transform(target)
                img = self.transform(img)

            img = self.noise_function(img).type(torch.float32)

        if self.give_index :
            return img, target, idx
        else :  
            return img, target


