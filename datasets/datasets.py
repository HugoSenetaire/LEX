import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
class FooDataset(Dataset):
    def __init__(self,shape = (3,3), len = 1000):
        # assert(shape is tuple)
        print(shape)
        # assert(len(shape)==2)
        self.size_x = shape[0]
        self.size_y = shape[1]
        self.nb_cat = self.size_x * self.size_y
        self.len = 1000
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ])
        

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        target = int(idx%self.nb_cat)
        x = np.ones((self.nb_cat), dtype=np.float32)
        x[target:] = np.zeros((self.nb_cat - target)) 
        x = x.reshape((self.size_x, self.size_y))

        data = self.transform(x)
        return data,target

class DatasetFoo():
    def __init__(self, batch_size_train, batch_size_test, shape = (3,3) , len = 1000):
        self.dataset = FooDataset(shape = shape, len = len)
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_train, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_test, shuffle=True)

    def get_category(self):
        return self.dataset.nb_cat

    def get_shape(self):
        return (1,self.dataset.size_x,self.dataset.size_y) 

class DatasetMnist():
    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader( torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('/files/', train=False, download=True,
                                                            transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                            ])),
                            batch_size=batch_size_test, shuffle=True
                            )

    def get_category(self):
        return 10

    def get_shape(self):
        return (1,28,28)