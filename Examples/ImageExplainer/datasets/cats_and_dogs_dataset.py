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


default_cat_dogs_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

catsanddogstreated = False
catsanddogstest = None
catsanddogstrain = None

def get_cats_and_dogs(path, test_ratio = 0.2):
    global catsanddogstest
    global catsanddogstrain
    train_files = os.listdir(path)
    catsanddogstrain, catsanddogstest = train_test_split(train_files, test_size = test_ratio)

class CatDogDataset(Dataset):
    def __init__(self,  root_dir = '/files/', train='train', download = False, transform = None, noisy = False, noise_function = None):
        self.dir = os.path.join(root_dir, "dogsvscats/train")

        global catsanddogstreated
        global catsanddogstrain
        global catsanddogstest

        if not catsanddogstreated :
            get_cats_and_dogs(self.dir)
            catsanddogstreated = True

        
        self.mode= train
        self.transform = transform
        if self.mode == "train": 
            self.file_list = catsanddogstrain
        else :
            self.file_list = catsanddogstest
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.file_list[idx].startswith("cat"):
            label = 0
        else :
            label = 1

        label = torch.tensor(label, dtype=torch.int64)
        

        if self.transform:
            img = self.transform(img)
            

        return img.type(torch.float32), label
