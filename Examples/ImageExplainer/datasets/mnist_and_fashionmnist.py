import torch
import torchvision 
import numpy as np 


from .dataset_from_data import DatasetFromData
from .mnist import default_MNIST_transform
from .utils import create_panels




class MNIST_and_FASHIONMNIST():
    def __init__(self,
            root_dir: str,
            transforms_mnist = default_MNIST_transform,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            target_mnist = True,
            random_panels = True,
            **kwargs,):

        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform = transforms_mnist)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform = transforms_mnist)
        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(root_dir, train=True, download=download, transform = transforms_mnist)
        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root_dir, train=False, download=download, transform = transforms_mnist)

        self.data_train_mnist = self.mnist_train.data /255.
        self.data_train_mnist += np.random.normal(0, 0.1, size = self.data_train_mnist.shape) #Handled the way it's handled in REAL X
        self.data_train_mnist = self.data_train_mnist.reshape(-1, 1, 28, 28)



        self.data_test_mnist = self.mnist_test.data /255.
        self.data_test_mnist += np.random.normal(0, 0.1, size = self.data_test_mnist.shape) #Handled the way it's handled in REAL X
        self.data_test_mnist = self.data_test_mnist.reshape(-1, 1, 28, 28)

        self.data_train_fashion = self.fashion_mnist_train.data /255.
        self.data_train_fashion += np.random.normal(0, 0.1, size = self.data_train_fashion.shape) #Handled the way it's handled in REAL X
        self.data_train_fashion = self.data_train_fashion.reshape(-1, 1, 28, 28)

        self.data_test_fashion = self.fashion_mnist_test.data /255.
        self.data_test_fashion += np.random.normal(0, 0.1, size = self.data_test_fashion.shape) #Handled the way it's handled in REAL X
        self.data_test_fashion = self.data_test_fashion.reshape(-1, 1, 28, 28)

        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets


        self.random_panels = random_panels
        target = "left" if target_mnist else "right"

        Xpanels_train, ypanels_train, self.quadrant_train = create_panels(self.data_train_mnist, self.data_train_fashion, self.mnist_train.targets, self.fashion_mnist_train.targets, random_panels=self.random_panels, target = target, )
        self.data_train = Xpanels_train
        self.target_train = ypanels_train

    

        Xpanels_test, ypanels_test, self.quadrant_test = create_panels(self.data_test_mnist, self.data_test_fashion, self.mnist_test.targets, self.fashion_mnist_test.targets, random_panels=self.random_panels, target = target, )
        self.data_test = Xpanels_test
        self.target_test = ypanels_test



        self.data_train = torch.tensor(self.data_train.reshape(-1,1,28,56), dtype = torch.float32, requires_grad=False)
        self.data_test = torch.tensor(self.data_test.reshape(-1,1,28,56), dtype = torch.float32, requires_grad=False)
        self.quadrant_test = torch.tensor(self.quadrant_test.reshape(-1,1,28,56), dtype = torch.int64, requires_grad=False)
        self.quadrant_train = torch.tensor(self.quadrant_train.reshape(-1,1,28,56), dtype = torch.int64, requires_grad=False)

        
        del self.mnist_train, self.mnist_test, self.fashion_mnist_train, self.fashion_mnist_test
        del self.data_train_mnist, self.data_test_mnist, self.data_train_fashion, self.data_test_fashion

        # TODO : DELETE THE ADDING OF NOISE
        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.optimal_S_train = self.quadrant_train
        self.optimal_S_test = self.quadrant_test



    def get_dim_input(self,):
        return (1,28,56)
        
    def get_dim_output(self,):
        return 10

    def get_true_selection(self, indexes, type = "test",):
        if type == "train" :
            optimal_S = self.optimal_S_train[indexes]
        elif type == "test" :
            optimal_S = self.optimal_S_test[indexes]
        else :
            raise ValueError("dataset_type must be either train or test")

        return optimal_S

    def __str__(self):
        return "Mnist_and_FashionMNIST"



