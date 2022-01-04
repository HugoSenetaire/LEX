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

np.random.seed(0)
torch.manual_seed(0)


default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])

class fromImageToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor):
        tensor = tensor.float()/255.
        return tensor


default_cat_dogs_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised



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


# ========================= AUGMENTED VISION DATASET ===============================
class DatasetFromData(Dataset):
    def __init__(self, data, target, transforms = None, target_transforms = None, give_index = False, noise_function = None) -> None:
        self.data = data
        self.targets = target
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.noise_function = noise_function
        self.give_index = give_index

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
        

## ======================= MNIST BASED DATASET ======================================
class MnistDataset():
    def __init__(self,
            root_dir: str,
            transform = default_MNIST_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):

        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, )
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, )

        self.data_train = self.mnist_train.data
        self.data_test = self.mnist_test.data
        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transform, target_transform, noise_function = noise_function)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transform, target_transform, noise_function = noise_function)

    def get_dim_input(self,):
        return (1,28,28)

    def get_dim_output(self,):
        return 10
# MNIST VARIATION :




# class MnistVariation1quadrant(MnistDataset):
#     def __init__(self,
#             root: str,
#             train: bool = True,
#             transform = None,
#             target_transform = None,
#             download: bool = False,
#             noisy: bool = False,
#             noise_function = None,
#             rate = 0.5,
#     ) :
#         super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
#         self.data_aux = []
#         middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
#         for element in self.data:
#             self.data_aux.append(element)
#             if np.random.random_sample((0))>rate :
#                 continue
#             index_new = np.random.randint(self.__len__())
#             img_new = self.data[index_new]

#             self.data_aux[-1][middle_size_x:, middle_size_y:] = img_new[middle_size_x:, middle_size_y:]
#         self.data = self.data_aux
           
#     def __str__(self):
#         return "MnistVariation1quadrant"

class DatasetFromData(Dataset):
    def __init__(self, data, target, transforms = None, target_transforms = None, give_index = False, noise_function = None) -> None:
        self.data = data
        self.targets = target
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.noise_function = noise_function
        self.give_index = give_index

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
      



class MNIST_and_FASHIONMNIST():
    def __init__(self,
            root_dir: str,
            transforms_mnist = default_MNIST_transform,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):
        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform = transforms_mnist)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform = transforms_mnist)
        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(root_dir, train=True, download=download, transform = transforms_mnist)
        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root_dir, train=False, download=download, transform = transforms_mnist)


        self.data_train_mnist = torch.zeros((len(self.mnist_train), 28, 28))
        for k in range(len(self.data_train_mnist)):
          self.data_train_mnist[k],_ = self.mnist_train.__getitem__(k) 

        self.data_test_mnist = torch.zeros((len(self.mnist_test), 28, 28))
        for k in range(len(self.data_test_mnist)):
          self.data_test_mnist[k], _ = self.mnist_test.__getitem__(k)

  
        self.data_train_fashion = torch.zeros((len(self.fashion_mnist_train), 28, 28))
        for k in range(len(self.data_train_fashion)):
          self.data_train_fashion[k],_ = self.fashion_mnist_train.__getitem__(k) 
        self.data_test_fashion = torch.zeros((len(self.fashion_mnist_test), 28, 28))
        for k in range(len(self.data_test_fashion)):
          self.data_test_fashion[k], _ = self.fashion_mnist_test.__getitem__(k)


        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets


        # Create the data :
        self.data_train = torch.zeros((len(self.data_train_mnist),28,56,))
        self.quadrant_train = torch.zeros((len(self.data_train_mnist),28,56))
        bernoulli_sample = np.random.binomial(1, 0.5, size = len(self.data_train_mnist))

        for k in range(len(self.data_train_mnist)):
            i = bernoulli_sample[k]
            j = 1-i
            self.data_train[k, :, i*28:(i+1)*28] = self.data_train_mnist[k]
            self.quadrant_train[k, :, i*28:(i+1)*28] = torch.ones((28,28))
            self.data_train[k, :, j*28:(j+1)*28] = self.data_train_fashion[k]
        

        # self.data_train.reshape((-1, 1, 28, 56))
        
        self.data_test = torch.zeros((len(self.data_test_mnist),28,56,))
        bernoulli_sample = np.random.binomial(1, 0.5, size = len(self.data_test_mnist))
        self.quadrant_test = torch.zeros((len(self.data_test_mnist),28,56))

        for k in range(len(self.data_test_mnist)):
            i = bernoulli_sample[k]
            j = 1-i
            self.data_test[k, :, i*28:(i+1)*28] = self.data_test_mnist[k]
            self.quadrant_test[k, :, i*28:(i+1)*28] = torch.ones((28,28))
            self.data_test[k, :, j*28:(j+1)*28] = self.data_test_fashion[k]
        
        # self.data_test.reshape((-1, 1, 28, 56))

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)

    def get_dim_input(self,):
        return (1,28,56)

    def get_dim_output(self,):
        return 10

    def __str__(self):
        return "Mnist_and_FashionMNIST"


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
    def __init__(self, dataset, batch_size_train = 16, batch_size_test=500, transform =None, noise_function = None,):
        self.dataset = dataset
        self.dataset_test = self.dataset.dataset_test
        self.dataset_train = self.dataset.dataset_train
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
     
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=batch_size_train, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=batch_size_test, shuffle=False)



