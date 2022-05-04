import torchvision 
import numpy as np
import torch
from .dataset_from_data import DatasetFromData
from .utils import create_panels

import medmnist
from medmnist import INFO, Evaluator


default_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                    #     (0.1307,), (0.3081,))
                                    ])

def get_no_nevi(data_class, shuffle = True, enforce_lenght = True):
    X =  data_class.imgs/255.
    X = np.transpose(X, (0,3,1,2))
    y =  data_class.labels.flatten()
    if shuffle :
        shuffled_index = np.arange(X.shape[0])
        np.random.shuffle(shuffled_index)
        X = X[shuffled_index]
        y = y[shuffled_index]

    index_nevi = np.where(y==5)[0]
    index_notnevi = np.where(y!=5)[0]
    
    data_nevi = X[index_nevi]
    y_nevi = np.zeros_like(y[index_nevi])
    data_notnevi = X[index_notnevi]
    y_notnevi = y[index_notnevi]
    y_notnevi[np.where(y_notnevi > 5)[0]] = y_notnevi[np.where(y_notnevi > 5)[0]] - 1 
    if enforce_lenght :
        lenght = min(len(data_nevi), len(data_notnevi))
        data_nevi = data_nevi[:lenght]
        y_nevi = y_nevi[:lenght]
        data_notnevi = data_notnevi[:lenght]
        y_notnevi = y_notnevi[:lenght]

    return data_nevi, y_nevi, index_nevi, data_notnevi, y_notnevi, index_notnevi
    

class Dermamnist():
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

        Xtrain =  self.dermamnist_train.imgs/255.
        Xtrain = np.transpose(Xtrain, (0,3,1,2))
        ytrain =  self.dermamnist_train.labels
       
        
        Xtest = self.dermamnist_test.imgs/255
        Xtest = np.transpose(Xtest, (0,3,1,2))
        ytest = self.dermamnist_test.labels
        

        self.data_train = torch.tensor(Xtrain, dtype=torch.float32)
        self.data_test = torch.tensor(Xtest, dtype=torch.float32)
        self.target_train = torch.tensor(ytrain, dtype=torch.long)
        self.target_test = torch.tensor(ytest, dtype=torch.long)

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)

    def get_dim_input(self,):
        return (3,28,28)

    def get_dim_output(self,):
        return 7





class DermamnistPanel():
    def __init__(self,
            root_dir: str,
            transform = default_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            random_panels = False,
            **kwargs,):

        self.random_panels = random_panels

        data_flag = 'dermamnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])


        # load the data
        self.dermamnist_train = DataClass(split='train',  download=download)
        self.dermamnist_test = DataClass(split='test',  download=download)


        

        #TRAIN DATASET
        data_train_nevi, y_train_nevi, index_train_nevi, data_train_notnevi, y_train_notnevi, index_train_notnevi = get_no_nevi(self.dermamnist_train, shuffle = True, enforce_lenght = True)
        Xpanels_train, ypanels_train, self.quadrant_train = create_panels(data_train_nevi, data_train_notnevi, y_train_nevi, y_train_notnevi, random_panels=self.random_panels, target = "right", )

        #TEST DATASET
        data_test_nevi, y_test_nevi, index_test_nevi, data_test_notnevi, y_test_notnevi, index_test_notnevi = get_no_nevi(self.dermamnist_test, shuffle = True, enforce_lenght = True)
        Xpanels_test, ypanels_test, self.quadrant_test = create_panels(data_test_nevi, data_test_notnevi, y_test_nevi, y_test_notnevi, random_panels=self.random_panels, target = "right", )

        
        
        self.data_train = torch.tensor(Xpanels_train, dtype=torch.float32)
        self.data_test = torch.tensor(Xpanels_test, dtype=  torch.float32)
        self.target_train = torch.tensor(ypanels_train, dtype=torch.long)
        self.target_test = torch.tensor(ypanels_test, dtype=torch.long)
        self.optimal_S_train = torch.tensor(self.quadrant_train, dtype=torch.float32)
        self.optimal_S_test = torch.tensor(self.quadrant_test, dtype=torch.float32)


        self.quadrant_train = np.zeros((self.data_train.shape[0], 1, *self.data_train.shape[2:]))
        self.quadrant_train[:, :, :, 28:] = np.ones_like(self.quadrant_train[:, :, :, 28:])
        self.quadrant_test = np.zeros((self.data_test.shape[0], 1, *self.data_test.shape[2:]))
        self.quadrant_test[:, :, :, 28:] = np.ones_like(self.quadrant_test[:, :, :, 28:])

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)

    def get_dim_input(self,):
        return (3,28,56)

    def get_dim_output(self,):
        return 6


