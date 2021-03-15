import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd


from sklearn import cluster, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


np.random.seed(0)
torch.manual_seed(0)


def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised


class TensorDatasetAugmented(TensorDataset):
    def __init__(self, x, y, noisy = False, noise_function = None):
        super().__init__(x,y)
        self.noisy = noisy
        self.noise_function = noise_function

    def __getitem__(self, idx):
        if not self.noisy :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx]
            return input_tensor, target
        else :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx].type(torch.float32)
            
            input_tensor = input_tensor.numpy()
            target = target.numpy()

            input_tensor = torch.tensor(self.noise_function(input_tensor)).type(torch.float32)
            return input_tensor, target

        

class CircleDataset():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train)
        self.data_test = torch.tensor(self.data_test)
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)


class CircleDatasetNotCentered():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, shift = [2,2], factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train + np.array(shift))
        self.data_test = torch.tensor(self.data_test + np.array(shift))
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)


##### ENCAPSULATION :

class LoaderArtificial():
    def __init__(self,dataset, batch_size_train = 1024, batch_size_test=1000, n_samples_train = 100000, n_samples_test=10000, noisy = False):

        self.dataset = dataset(n_samples_train = n_samples_train, n_samples_test = n_samples_test, noisy = noisy)
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader( self.dataset_train,
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                self.dataset_test,
                            batch_size=batch_size_test, shuffle=False
                            )

    def get_category(self):
        return 2

    def get_shape(self):
        return (2)

