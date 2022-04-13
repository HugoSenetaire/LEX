import torch
import pickle as pkl
import os

from .dataset_from_data import DatasetFromData


class MNISTImageBackground():
    def __init__(self,
            root_dir: str,
            transforms_mnist = None,
            target_transforms = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):

        self.optimal_S_train = None
        self.optimal_S_test = None
        
        self.ground_truth = False

        path_train = os.path.join(root_dir, "mnist_background_images_train.pkl")
        path_test = os.path.join(root_dir, "mnist_background_images_test.pkl")

        with open(path_train, "rb") as f :
            self.data_test, self.target_test = pkl.load(f) 

        with open(path_test, "rb") as f :
            self.data_train, self.target_train = pkl.load(f)

        self.data_train = torch.tensor(self.data_train, dtype = torch.float32)
        self.data_test = torch.tensor(self.data_test, dtype = torch.float32)
        self.target_train = torch.tensor(self.target_train, dtype= torch.int64)
        self.target_test = torch.tensor(self.target_test, dtype= torch.int64)

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = target_transforms, noise_function = noise_function, give_index=True)

    def get_dim_input(self,):
        return (1,28,28)

    def get_dim_output(self,):
        return 10

    def __str__(self):
        return "MNISTImageBackground"
