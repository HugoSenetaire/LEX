import torchvision 
import numpy as np
import torch
from .dataset_from_data import DatasetFromData

import medmnist
from medmnist import INFO, Evaluator


default_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                    #     (0.1307,), (0.3081,))
                                    ])


class DermamnistLeftPanel():
    def __init__(self,
            root_dir: str,
            transform = default_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            **kwargs,):


        data_flag = 'dermamnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])


        # load the data
        self.dermamnist_train = DataClass(split='train',  download=download)
        self.dermamnist_test = DataClass(split='test',  download=download)

        Xtrain =  self.dermamnist_train.imgs/255
        Xtrain = np.transpose(Xtrain, (0,3,1,2))
        np.random.shuffle(Xtrain)
        ytrain =  self.dermamnist_train.labels

        nevi_train = Xtrain[np.where(ytrain==5)[0],] # indices of nevi_train
        notnevi_train = Xtrain[np.where(ytrain!=5)[0],]  #indices of non-nevi_train
        nevi_train = nevi_train[:np.sum(ytrain!=5),:] # we extract the first 2314 nevi_train
        Xpanels_train = np.zeros((nevi_train.shape[0], 3, 28, 28*2))
        Xpanels_train[:, :, :, :28,] = nevi_train
        Xpanels_train[:, :, :, 28:,] = notnevi_train

        ypanels_train = ytrain[np.where(ytrain!=5)[0]]
        ypanels_train[np.where(ypanels_train>5)[0]] = ypanels_train[np.where(ypanels_train>5)[0]] - 1

        
        Xtest = self.dermamnist_test.imgs/255
        Xtest = np.transpose(Xtest, (0,3,1,2))
        np.random.shuffle(Xtest)
        ytest = self.dermamnist_test.labels


        nevi_test = Xtest[np.where(ytest==5)[0],] # indices of nevi_test
        notnevi_test = Xtest[np.where(ytest!=5)[0],]  #indices of non-nevi_test
        nevi_test = nevi_test[:np.sum(ytest!=5),:] # we extract the first 2314 nevi
        Xpanels_test = np.zeros((nevi_test.shape[0], 3, 28, 28*2))
        Xpanels_test[:, :, :, 0:28,] = nevi_test
        Xpanels_test[:, :, :, 28:,] = notnevi_test


        ypanels_test = ytest[np.where(ytest!=5)[0]]
        ypanels_test[np.where(ypanels_test>5)[0]] = ypanels_test[np.where(ypanels_test>5)[0]] - 1

        self.data_train = torch.tensor(Xpanels_train.reshape(-1,3,28,56), dtype=torch.float32)
        self.data_test = torch.tensor(Xpanels_test.reshape(-1,3,28,56), dtype=  torch.float32)
        self.target_train = torch.tensor(ypanels_train, dtype=torch.long)
        self.target_test = torch.tensor(ypanels_test, dtype=torch.long)
        

        self.quadrant_train = np.zeros((self.data_train.shape[0], 1, *self.data_train.shape[2:]))
        self.quadrant_train[:, :, :, 28:] = np.ones_like(self.quadrant_train[:, :, :, 28:])
        self.quadrant_test = np.zeros((self.data_test.shape[0], 1, *self.data_test.shape[2:]))
        self.quadrant_test[:, :, :, 28:] = np.ones_like(self.quadrant_test[:, :, :, 28:])

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.optimal_S_train = torch.tensor(self.quadrant_train, dtype=torch.float32)
        self.optimal_S_test = torch.tensor(self.quadrant_test, dtype=torch.float32)

    def get_dim_input(self,):
        return (3,28,56)

    def get_dim_output(self,):
        return 6


