import torch
import torchvision 
import numpy as np 


from .dataset_from_data import DatasetFromData
from .mnist import default_MNIST_transform




class MNIST_and_FASHIONMNIST():
    def __init__(self,
            root_dir: str,
            transforms_mnist = default_MNIST_transform,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):

        self.ground_truth_selection = True
        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform = transforms_mnist)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform = transforms_mnist)
        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(root_dir, train=True, download=download, transform = transforms_mnist)
        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root_dir, train=False, download=download, transform = transforms_mnist)


        self.data_train_mnist = torch.zeros((len(self.mnist_train), 28, 28))
        for k in range(len(self.data_train_mnist)):
          self.data_train_mnist[k],_ = self.mnist_train.__getitem__(k) 
        # self.data_train_mnist = self.data_train_mnist/torch.tensor(255.)

        self.data_test_mnist = torch.zeros((len(self.mnist_test), 28, 28))
        for k in range(len(self.data_test_mnist)):
          self.data_test_mnist[k], _ = self.mnist_test.__getitem__(k)
        # self.data_test_mnist = self.data_test_mnist/torch.tensor(255.)

  
        self.data_train_fashion = torch.zeros((len(self.fashion_mnist_train), 28, 28))
        for k in range(len(self.data_train_fashion)):
          self.data_train_fashion[k],_ = self.fashion_mnist_train.__getitem__(k) 
        # self.data_train_fashion = self.data_train_fashion/torch.tensor(255.)

        self.data_test_fashion = torch.zeros((len(self.fashion_mnist_test), 28, 28))
        for k in range(len(self.data_test_fashion)):
          self.data_test_fashion[k], _ = self.fashion_mnist_test.__getitem__(k)
        # self.data_test_fashion = self.data_test_fashion/torch.tensor(255.)



        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets

        self.quadrant_filling = torch.ones_like(self.data_train_mnist[0, :, :])
        self.quadrant_filling = torch.where(torch.std(self.data_train_mnist, dim = 0) == 0, torch.zeros_like(self.quadrant_filling), self.quadrant_filling)        

        #TODO : DELETE THIS
        # self.data_train_mnist = self.data_train_mnist[:10]
        # self.data_test_mnist = self.data_test_mnist[:10]

        # Create the data :
        self.data_train = torch.zeros((len(self.data_train_mnist),28,56,))
        self.quadrant_train = torch.zeros((len(self.data_train_mnist),28,56))
        bernoulli_sample = np.random.binomial(1, 0.5, size = len(self.data_train_mnist))

        for k in range(len(self.data_train_mnist)):
            i = bernoulli_sample[k]
            j = 1-i
            self.data_train[k, :, i*28:(i+1)*28] = self.data_train_mnist[k]
            self.quadrant_train[k, :, i*28:(i+1)*28] = self.quadrant_filling
            self.data_train[k, :, j*28:(j+1)*28] = self.data_train_fashion[k]
        
        
        self.data_test = torch.zeros((len(self.data_test_mnist),28,56,))
        bernoulli_sample = np.random.binomial(1, 0.5, size = len(self.data_test_mnist))
        self.quadrant_test = torch.zeros((len(self.data_test_mnist),28,56))

        for k in range(len(self.data_test_mnist)):
            i = bernoulli_sample[k]
            j = 1-i
            self.data_test[k, :, i*28:(i+1)*28] = self.data_test_mnist[k]
            self.quadrant_test[k, :, i*28:(i+1)*28] = self.quadrant_filling
            self.data_test[k, :, j*28:(j+1)*28] = self.data_test_fashion[k]

        

        # TODO : DELETE THE ADDING OF NOISE
        # self.data_train += torch.normal(0, 0.2, size = self.data_train.shape)
        # self.data_test += torch.normal(0, 0.2, size = self.data_test.shape)
        # self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        # self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        # self.optimal_S_train = self.quadrant_train
        # self.optimal_S_test = self.quadrant_test


    def get_dim_input(self,):
        return (1,28,56)
        
    def get_dim_output(self,):
        return 10


    def __str__(self):
        return "Mnist_and_FashionMNIST"






class FASHIONMNIST_and_MNIST():
    def __init__(self,
            root_dir: str,
            transforms_mnist = default_MNIST_transform,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):

        self.ground_truth_selection = True
        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform = transforms_mnist)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform = transforms_mnist)
        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(root_dir, train=True, download=download, transform = transforms_mnist)
        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root_dir, train=False, download=download, transform = transforms_mnist)


        self.data_train_mnist = torch.zeros((len(self.mnist_train), 28, 28))
        for k in range(len(self.data_train_mnist)):
          self.data_train_mnist[k],_ = self.mnist_train.__getitem__(k) 
        # self.data_train_mnist = self.data_train_mnist/torch.tensor(255.)

        self.data_test_mnist = torch.zeros((len(self.mnist_test), 28, 28))
        for k in range(len(self.data_test_mnist)):
          self.data_test_mnist[k], _ = self.mnist_test.__getitem__(k)
        # self.data_test_mnist = self.data_test_mnist/torch.tensor(255.)

  
        self.data_train_fashion = torch.zeros((len(self.fashion_mnist_train), 28, 28))
        for k in range(len(self.data_train_fashion)):
          self.data_train_fashion[k],_ = self.fashion_mnist_train.__getitem__(k) 
        # self.data_train_fashion = self.data_train_fashion/torch.tensor(255.)

        self.data_test_fashion = torch.zeros((len(self.fashion_mnist_test), 28, 28))
        for k in range(len(self.data_test_fashion)):
          self.data_test_fashion[k], _ = self.fashion_mnist_test.__getitem__(k)
        # self.data_test_fashion = self.data_test_fashion/torch.tensor(255.)



        self.target_train = self.fashion_mnist_train.targets
        self.target_test = self.fashion_mnist_test.targets
        

        #TODO : DELETE THIS
        # self.data_train_mnist = self.data_train_mnist[:10]
        # self.data_test_mnist = self.data_test_mnist[:10]

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
        
        
        
        self.data_test = torch.zeros((len(self.data_test_mnist),28,56,))
        bernoulli_sample = np.random.binomial(1, 0.5, size = len(self.data_test_mnist))
        self.quadrant_test = torch.zeros((len(self.data_test_mnist),28,56))

        for k in range(len(self.data_test_mnist)):
            i = bernoulli_sample[k]
            j = 1-i
            self.data_test[k, :, i*28:(i+1)*28] = self.data_test_mnist[k]
            self.quadrant_test[k, :, i*28:(i+1)*28] = torch.ones((28,28))
            self.data_test[k, :, j*28:(j+1)*28] = self.data_test_fashion[k]

        

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.optimal_S_train = self.quadrant_train
        self.optimal_S_test = self.quadrant_test

    def get_dim_input(self,):
        return (1,28,56)

    def get_dim_output(self,):
        return 10

    def __str__(self):
        return "FashionMNIST_and__Mnist"
