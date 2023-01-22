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

def getwhitecell(data_class, shuffle = False, enforce_lenght = True):
    
    X =  data_class.imgs/255.
    X = np.transpose(X, (0,3,1,2))
    y =  data_class.labels
    if shuffle :
        shuffled_index = np.arange(X.shape[0])
        np.random.shuffle(shuffled_index)
        X = X[shuffled_index]
        y = y[shuffled_index]

    index_whitecell = np.where((y!= 3) & (y!=7))[0]
    index_notwhitecell = np.where((y== 3) | (y==7))[0]
    

    data_whitecell = X[index_whitecell]
    y_whitecell = y[index_whitecell]
    y_whitecell[np.where(y_whitecell > 7)[0]] = y_whitecell[np.where(y_whitecell > 7)[0]] - 1
    y_whitecell[np.where(y_whitecell > 3)[0]] = y_whitecell[np.where(y_whitecell > 3)[0]] - 1
    data_notwhitecell = X[index_notwhitecell]
    y_notwhitecell = y[index_notwhitecell]
    y_notwhitecell[np.where(y_notwhitecell == 7)[0]] = 1
    y_notwhitecell[np.where(y_notwhitecell == 3)[0]] = 0

    if enforce_lenght :
        lenght = min(len(data_whitecell), len(data_notwhitecell))
        data_whitecell = data_whitecell[:lenght]
        y_whitecell = y_whitecell[:lenght]
        data_notwhitecell = data_notwhitecell[:lenght]
        y_notwhitecell = y_notwhitecell[:lenght]

    return data_whitecell, y_whitecell, index_whitecell, data_notwhitecell, y_notwhitecell, index_notwhitecell



class BloodMNIST():
    def __init__(self,
            root_dir: str,
            transform = default_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            give_index = False,
            **kwargs,):


        data_flag = 'bloodmnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        self.give_index = give_index
        DataClass = getattr(medmnist, info['python_class'])


        # load the data
        self.bloodmnist_train = DataClass(split='train',  download=download)
        Xtrain =  self.bloodmnist_train.imgs/255.
        Xtrain = np.transpose(Xtrain, (0,3,1,2))
        ytrain =  self.bloodmnist_train.labels

        self.bloodmnist_test = DataClass(split='test',  download=download)
        Xtest = self.bloodmnist_test.imgs/255
        Xtest = np.transpose(Xtest, (0,3,1,2))
        ytest = self.bloodmnist_test.labels

        self.bloodmnist_val = DataClass(split='val',  download=download)
        Xval = self.bloodmnist_val.imgs/255
        Xval = np.transpose(Xval, (0,3,1,2))
        yval = self.bloodmnist_val.labels
        

        self.data_train = torch.tensor(Xtrain, dtype=torch.float32)
        self.data_test = torch.tensor(Xtest, dtype=torch.float32)
        self.data_val = torch.tensor(Xval, dtype=torch.float32)
        self.target_train = torch.tensor(ytrain, dtype=torch.long)
        self.target_test = torch.tensor(ytest, dtype=torch.long)
        self.target_val = torch.tensor(yval, dtype=torch.long)

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_val = DatasetFromData(self.data_val, self.target_val, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)

    def get_dim_input(self,):
        return (3,28,28)

    def get_dim_output(self,):
        return 8


class WhiteCellMNIST():
    def __init__(self,
            root_dir: str,
            transform = default_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            give_index = False,
            **kwargs,):

        self.give_index = give_index
        data_flag = 'bloodmnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])


        # load the data
        self.bloodmnist_train = DataClass(split='train',  download=download)
        self.bloodmnist_test = DataClass(split='test',  download=download)
        self.bloodmnist_val = DataClass(split='val',  download=download)

        
        whitecell_train, y_whitecell_train, index_whitecell_train, notwhitecell_train, y_notwhitecell_train, index_notwhitecell_train = getwhitecell(self.bloodmnist_train, shuffle = True, enforce_lenght=False)
        Xtrain = whitecell_train
        ytrain = y_whitecell_train

        
        whitecell_test, y_whitecell_test, index_whitecell_test, notwhitecell_test, y_notwhitecell_test, index_notwhitecell_test = getwhitecell(self.bloodmnist_test, shuffle = False, enforce_lenght=False)
        Xtest = whitecell_test
        ytest = y_whitecell_test

        whitecell_val, y_whitecell_val, index_whitecell_val, notwhitecell_val, y_notwhitecell_val, index_notwhitecell_val = getwhitecell(self.bloodmnist_val, shuffle = False, enforce_lenght=False)
        Xval = whitecell_val
        yval = y_whitecell_val


        self.data_train = torch.tensor(Xtrain, dtype=torch.float32)
        self.data_test = torch.tensor(Xtest, dtype=torch.float32)
        self.data_val = torch.tensor(Xval, dtype=torch.float32)
        self.target_train = torch.tensor(ytrain, dtype=torch.long)
        self.target_test = torch.tensor(ytest, dtype=torch.long)
        self.target_val = torch.tensor(yval, dtype=torch.long)

        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=self.give_index)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=self.give_index)
        self.dataset_val = DatasetFromData(self.data_val, self.target_val, transforms = None, target_transforms = None, noise_function = noise_function, give_index=self.give_index)

    def get_dim_input(self,):
        return (3,28,28)

    def get_dim_output(self,):
        return 6





class BloodMNISTPanel():
    def __init__(self,
            root_dir: str,
            transform = default_transform,
            target_transform = None,
            download: bool = False,
            noise_function = None,
            target_whitecell = False,
            random_panel = False,
            give_index = False,
            **kwargs,):

        self.random_panel = random_panel
        data_flag = 'bloodmnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        self.give_index = give_index

        DataClass = getattr(medmnist, info['python_class'])
        self.target_whitecell = target_whitecell
    


        # load the data
        self.bloodmnist_train = DataClass(split='train',  download=download)
        self.bloodmnist_test = DataClass(split='test',  download=download)
        self.bloodmnist_val = DataClass(split='val',  download=download)
        target = "left" if target_whitecell else "right"
       
        # TRAIN DATASET
        whitecell_train, y_whitecell_train, index_whitecell_train, notwhitecell_train, y_notwhitecell_train, index_notwhitecell_train = getwhitecell(self.bloodmnist_train, shuffle = True, enforce_lenght=True)
        Xpanels_train, ypanels_train, self.quadrant_train = create_panels(whitecell_train, notwhitecell_train, y_whitecell_train, y_notwhitecell_train, random_panels=self.random_panel, target = target, )


        # TEST DATASET
        whitecell_test, y_whitecell_test, index_whitecell_test, notwhitecell_test, y_notwhitecell_test, index_notwhitecell_test = getwhitecell(self.bloodmnist_test, shuffle = False, enforce_lenght=True)
        Xpanels_test, ypanels_test, self.quadrant_test = create_panels(whitecell_test, notwhitecell_test, y_whitecell_test, y_notwhitecell_test, random_panels=self.random_panel, target = target, )

        # VAL DATASET
        whitecell_val, y_whitecell_val, index_whitecell_val, notwhitecell_val, y_notwhitecell_val, index_notwhitecell_val = getwhitecell(self.bloodmnist_val, shuffle = False, enforce_lenght=True)
        Xpanels_val, ypanels_val, self.quadrant_val = create_panels(whitecell_val, notwhitecell_val, y_whitecell_val, y_notwhitecell_val, random_panels=self.random_panel, target = target, )

        self.data_train = torch.tensor(Xpanels_train.reshape(-1,3,28,56), dtype = torch.float32)
        self.data_test = torch.tensor(Xpanels_test.reshape(-1,3,28,56), dtype = torch.float32)
        self.data_val = torch.tensor(Xpanels_val.reshape(-1,3,28,56), dtype = torch.float32)
        self.target_train = torch.tensor(ypanels_train, dtype=torch.long)
        self.target_test = torch.tensor(ypanels_test, dtype=torch.long)
        self.target_val = torch.tensor(ypanels_val, dtype=torch.long)


        self.dataset_train = DatasetFromData(self.data_train, self.target_train, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_test = DatasetFromData(self.data_test, self.target_test, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.dataset_val = DatasetFromData(self.data_val, self.target_val, transforms = None, target_transforms = None, noise_function = noise_function, give_index=True)
        self.optimal_S_train = torch.tensor(self.quadrant_train, dtype=torch.float32)
        self.optimal_S_test = torch.tensor(self.quadrant_test, dtype=torch.float32)
        self.optimal_S_val = torch.tensor(self.quadrant_val, dtype=torch.float32)

    def get_true_selection(self, indexes, type = "test",):
        if type == "train" :
            optimal_S = self.optimal_S_train[indexes]
        elif type == "test" :
            optimal_S = self.optimal_S_test[indexes]
        else :
            raise ValueError("dataset_type must be either train or test")

        return optimal_S

    def get_dim_input(self,):
        return (3,28,56)

    def get_dim_output(self,):
        if self.target_whitecell :
            return 6
        else :
            return 2

