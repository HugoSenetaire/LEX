import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd

np.random.seed(0)
torch.manual_seed(0)


default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])

def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised


class FooDataset(Dataset):
    def __init__(self,shape = (3,3), len_dataset = 10000, shift = 3):
        self.size_x = shape[0]
        self.size_y = shape[1]
        self.nb_cat = self.size_x * self.size_y
        self.shift = shift 
        self.len = len_dataset
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ])
        

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        target = int(idx%(self.nb_cat-self.shift))
        x = np.ones((self.nb_cat), dtype=np.float64)
        x[(target+1):] = np.zeros((self.nb_cat - (target+1))) 
        x = x.reshape((self.size_x, self.size_y))

        data = self.transform(x)
        return data,target

class DatasetFoo():
    def __init__(self, batch_size_train, batch_size_test, shape = (3,3) , len_dataset = 10000):
        self.dataset = FooDataset(shape = shape, len_dataset = len_dataset)
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_train, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_test, shuffle=True)

    def get_category(self):
        return self.dataset.nb_cat - self.dataset.shift

    def get_shape(self):
        return (1,self.dataset.size_x,self.dataset.size_y) 


## ======================= MNIST ======================================



# MNIST VARIATION :


class MnistDataset(torchvision.datasets.MNIST):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download)
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

            return img, target
        else :
            img, target = self.data[idx], self.data[idx]
            
            img = img.numpy()
            target = target.numpy()
      
            if self.transform is not None:
                target = self.transform(target)
                img = self.transform(img)

            img = self.noise_function(img).type(torch.float32)


            return img, target


class MnistVariation1(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.data_aux = []
        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        for element in self.data:
            index_new = np.random.randint(self.__len__())
            img_new = self.data[index_new]
            self.data_aux.append(element)
            self.data_aux[-1][middle_size_x:, middle_size_y:] = img_new[middle_size_x:, middle_size_y:]
        self.data = self.data_aux
           
    def __str__(self):
        return "MnistVariation1"
        

class MnistVariation1quadrant(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            rate = 0.5,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.data_aux = []
        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        for element in self.data:
            self.data_aux.append(element)
            if np.random.random_sample((0))>rate :
                continue
            index_new = np.random.randint(self.__len__())
            img_new = self.data[index_new]

            self.data_aux[-1][middle_size_x:, middle_size_y:] = img_new[middle_size_x:, middle_size_y:]
        self.data = self.data_aux
           
    def __str__(self):
        return "MnistVariation1quadrant"
        

class MnistVariation2(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.data_aux = copy.deepcopy(self.data)
        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        for k in range(len(self.data)):
            index_new = np.random.randint(self.__len__())
            img_next, target_next = self.data[index_new], int(self.targets[index_new])
            if target_next > target :
                self.targets[k] = target_next
            self.data_aux[k][middle_size_x:, :] = img_next[middle_size_x:, :]

        self.data = self.data_aux
    def __str__(self):
        return "MnistVariation2"



class MnistVariationFashion(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.fashion_mnist = torchvision.datasets.FashionMNIST(root, train, transform, target_transform, download)
        self.data_aux = copy.deepcopy(self.data)
        self.fashion_data = self.fashion_mnist.data

        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        quadrant_x = np.random.randint(2, size = (len(self.data)))
        quadrant_y = np.random.randint(2, size = (len(self.data)))
        anchor_x_1 = middle_size_x *quadrant_x
        anchor_x_2 = anchor_x_1 + middle_size_x
        anchor_y_1 = middle_size_y * quadrant_y
        anchor_y_2 = anchor_y_1 + middle_size_y
        for k in range(len(self.data)):
            index_new = np.random.randint(len(self.fashion_data))
            img_next = self.fashion_data[index_new]
            self.data_aux[k][anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]] = img_next[anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]]

        self.data = self.data_aux
    def __str__(self):
        return "MnistVariation2"

class MnistVariationFashion2(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            rate = 0.5,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.fashion_mnist = torchvision.datasets.FashionMNIST(root, train, transform, target_transform, download)
        self.data_aux = copy.deepcopy(self.data)
        self.fashion_data = self.fashion_mnist.data

        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        quadrant_x = np.random.randint(2, size = (len(self.data)))
        quadrant_y = np.random.randint(2, size = (len(self.data)))
        anchor_x_1 = middle_size_x *quadrant_x
        anchor_x_2 = anchor_x_1 + middle_size_x
        anchor_y_1 = middle_size_y * quadrant_y
        anchor_y_2 = anchor_y_1 + middle_size_y
        for k in range(len(self.data)):
            if np.random.random_sample((0))>rate :
                continue
            index_new = np.random.randint(len(self.fashion_data))
            img_next = self.fashion_data[index_new]
            self.data_aux[k][anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]] = img_next[anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]]

        self.data = self.data_aux
    def __str__(self):
        return "MnistFashionVariation2"

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
    ) :
        super().__init__(root, train, transform, target_transform, download)
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

            return img, target
        else :
            img, target = self.data[idx], self.data[idx]
            
            img = img.numpy()
            target = target.numpy()
      
            if self.transform is not None:
                target = self.transform(target)
                img = self.transform(img)

            img = self.noise_function(img).type(torch.float32)


            return img, target


# MIX MNIST OMNIGLOT

##### ENCAPSULATION :

class LoaderEncapsulation():
    def __init__(self, dataset_class = MnistDataset, batch_size_train = 64, batch_size_test=1000, transform = default_MNIST_transform, noisy = False, noise_function = None ):
        self.dataset_class = dataset_class
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
     
        self.train_loader = torch.utils.data.DataLoader( dataset_class('/files/', train=True, download=True,
                                transform=transform, noisy=noisy, noise_function=noise_function),
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                dataset_class('/files/', train=False, download=True,
                                    transform=transform, noisy=noisy, noise_function=noise_function
                                                            ),
                            batch_size=batch_size_test, shuffle=False
                            )

    def get_category(self):
        return 10

    def get_shape(self):
        return (1,28,28)

