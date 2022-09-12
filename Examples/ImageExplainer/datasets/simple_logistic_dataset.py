import torch
import torchvision 
import numpy as np 


from .dataset_from_data import DatasetFromData
from .mnist import default_MNIST_transform
from .utils import create_panels




class SimpleLogisticDataset():
    def __init__(self,
            root_dir: str,
            transforms_mnist = default_MNIST_transform,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            target_mnist = True,
            random_panels = True,
            **kwargs,):


        size = 10000
        self.data =  np.zeros((size, 1, 2, 2))
        self.data[:,0,0,0] = 0
        self.data[:,0,0,1] = 255
        random_index = np.random.choice(np.arange(size), size=size//2, replace=False)
        opposite_random_index = np.setdiff1d(np.arange(size), random_index)

        self.data[random_index,0,1,0] = np.round(np.random.uniform(0,100,size//2))
        self.data[random_index,0,1,1] = np.round(np.random.uniform(100,200,size//2))
        self.data[opposite_random_index,0,1,0] = np.round(np.random.uniform(150,250,size//2))
        self.data[opposite_random_index,0,1,1] = np.round(np.random.uniform(0,100,size//2))
        self.data = torch.tensor(self.data, dtype = torch.float32)
        self.data_train = self.data[:int(size*0.8)]
        self.data_test = self.data[int(size*0.8):]
        self.target_train = torch.tensor(np.zeros(int(size * 0.8)), dtype = torch.long)
        self.target_test = torch.tensor(np.ones(int(size * 0.2)), dtype = torch.long)






        self.data_train = torch.tensor(self.data_train.reshape(-1,1,2,2,), dtype = torch.float32, requires_grad=False)
        self.data_test = torch.tensor(self.data_test.reshape(-1,1,2,2,), dtype = torch.float32, requires_grad=False)





        # TODO : DELETE THE ADDING OF NOISE
        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)



    def get_dim_input(self,):
        return (1,2,2,)
        
    def get_dim_output(self,):
        return 2

    def __str__(self):
        return "Mnist_and_FashionMNIST"



