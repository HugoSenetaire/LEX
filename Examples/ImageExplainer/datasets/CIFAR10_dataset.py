
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

default_CIFAR10_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((224,224),),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

class CIFAR10():
    def __init__(self,
            root_dir: str,
            transform = default_CIFAR10_transform,
            target_transform = None,
            download: bool = True,
            random_seed = 0,
            **kwargs,):

        self.CIFAR10_train = torchvision.datasets.CIFAR10(root = root_dir, train=True, download=download, transform=transform)
        self.CIFAR10_test  = torchvision.datasets.CIFAR10(root = root_dir, train=False, download=download, transform=transform)

        self.dataset_train = self.CIFAR10_train
        self.dataset_test = self.CIFAR10_test

    def get_dim_input(self,):
        return (3,224,224)

    def get_dim_output(self,):
        return 10

    


