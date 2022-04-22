import torchvision 
import numpy as np
import torch
from .dataset_from_data import DatasetFromData

default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                    #     (0.1307,), (0.3081,))
                                    ])



class MnistDataset():
    def __init__(self,
            root_dir: str,
            transform = default_MNIST_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):

        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform=default_MNIST_transform)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform=default_MNIST_transform)

        self.data_train = self.mnist_train.data / 255.
        self.data_train += np.random.normal(0, 0.1, size = self.data_train.shape) #Handled the way it's handled in REAL X
        self.data_test = self.mnist_test.data / 255.
        self.data_test += np.random.normal(0, 0.1, size = self.data_test.shape) #Handled the way it's handled in REAL X
        self.data_train = self.data_train.type(torch.float32)
        self.data_test = self.data_test.type(torch.float32)
        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets

        self.data_train = self.data_train.reshape(-1,1,28,28)
        self.data_test = self.data_test.reshape(-1,1,28,28)
        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transform, target_transform, noise_function = noise_function)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transform, target_transform, noise_function = noise_function)

    def get_dim_input(self,):
        return (1,28,28)

    def get_dim_output(self,):
        return 10



      