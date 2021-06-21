from numpy.core.fromnumeric import nonzero
import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd


from sklearn import cluster, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice




sequence_len = 700
total_features = 57
amino_acid_residues = 22


class TensorDatasetAugmented(TensorDataset):
    def __init__(self, x, y, noisy = False, noise_function = None):
        print(y[0].size)
        super().__init__(x,y)
        self.noisy = noisy
        self.noise_function = noise_function

    def __getitem__(self, idx):
        if not self.noisy :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx]
            return input_tensor, target
        else :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx].type(torch.float32)
            
            input_tensor = input_tensor.numpy()
            target = target.numpy()

            input_tensor = torch.tensor(self.noise_function(input_tensor)).type(torch.float32)
            return input_tensor, target



def split_with_shuffle(Dataset, seed=0):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0]*0.8)
    test_val_split = int(Dataset.shape[0]*0.1)
    Train = Dataset[0:train_split, :, :]
    Test = Dataset[train_split:train_split+test_val_split, :, :]
    Validation = Dataset[train_split+test_val_split:, :, :]
    return Train, Test, Validation

class cullpdb_6133_8classes():
    def __init__(self,
            root: str,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            cnn_width = 19,
    ) :
        if noisy == True :
            assert(noise_function is not None)
        self.cnn_width = cnn_width
        self.num_classes = 8
        path=os.path.join(root, "cullpdb+profile_6133.npy")
        path = "D:\scratch\hhjs\dataset\cullpdb+profile_6133.npy"
        path_aux = "D:\scratch\hhjs\dataset\cullpdb_reducedv2.npy"
        path_aux = "D:\scratch\hhjs\dataset\cullpdb_reduced.npy"
        if not os.path.exists(path):
            if download == True :
                raise NotImplementedError('The download function is not yet implemented') # TODO
            else :
                raise NotImplementedError('The data has not been downloaded, should use download argument') # TODO : Change the exception
        self.noisy = noisy
        self.noise_function = noise_function
        ds = np.load(path_aux)
        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))

        # ds2 = ds[:1000,:,:30]
        # np.save(path_aux, ds2)
        # assert(1==0)
        
    
        ret = np.zeros((ds.shape[0], ds.shape[1], amino_acid_residues + self.num_classes),dtype=np.int32)
        ret[:, :, :amino_acid_residues] = ds[:, :, :amino_acid_residues]
        ret[:, :, amino_acid_residues:] = ds[:, :, amino_acid_residues :amino_acid_residues+ self.num_classes]
    

        # print(ret.shape)
        D_train, D_test, D_val = split_with_shuffle(ret)

        self.X_train, self.Y_train = self.get_data_labels(D_train)
        # print(self.X_train.shape)
        # print(np.where(self.X_train>0)[1])
        self.X_test, self.Y_test = self.get_data_labels(D_test)
        self.X_val, self.Y_val = self.get_data_labels(D_val)


        self.X_train = torch.tensor(self.reshape_data(self.X_train))
        self.X_test = torch.tensor(self.reshape_data(self.X_test))
        self.X_val = torch.tensor(self.reshape_data(self.X_val))

        self.Y_train = torch.tensor(self.reshape_labels(self.Y_train), dtype = torch.int64)
        self.Y_train = torch.argmax(self.Y_train, axis = 1)
        self.Y_test = torch.tensor(self.reshape_labels(self.Y_test), dtype = torch.int64)
        self.Y_test = torch.argmax(self.Y_test, axis = 1)
        self.Y_val = torch.tensor(self.reshape_labels(self.Y_val), dtype = torch.int64)

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function)

    def reshape_data(self, X):
        padding = np.zeros((X.shape[0], X.shape[2], int(self.cnn_width/2)))
        # print(np.shape(X))
        X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
        # print(np.shape(X))
        # print(np.shape(padding))

        X = np.swapaxes(X, 1, 2)
        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues))
        for i in range(X.shape[1] - self.cnn_width + 1):
            # print(i, np.count_nonzero(X[0, i:i+self.cnn_width, :]), np.where(X[0, i:i+self.cnn_width]>0)[0])
            res[:, i, :, :] = X[:, i:i+self.cnn_width, :]
        res = np.reshape(res, (X.shape[0]*(X.shape[1] - self.cnn_width + 1),amino_acid_residues, self.cnn_width))
        # print("non zero 2", np.count_nonzero(res))
        # for k in range(20):
        #     print(np.count_nonzero(res[k]))
        #     print(res[k])
        # res = res[np.count_nonzero(res, axis=(1,2))>(int(self.cnn_width/2)*amino_acid_residues), :, :]
        # print(res.shape)
        return res


    def reshape_labels(self, labels):
        Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))
        # Y = Y[~np.all(Y == 0, axis=1)]
        return Y


    def get_data_labels(self, D):
        X = D[:, :, 0:amino_acid_residues]
        Y = D[:, :, amino_acid_residues:amino_acid_residues + self.num_classes]
        return X, Y
    

class cullpdb_6133_8classes_nosides():
    def __init__(self,
            root: str,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            cnn_width = 19,
    ) :
        if noisy == True :
            assert(noise_function is not None)
        self.cnn_width = cnn_width
        self.num_classes = 8
        path=os.path.join(root, "cullpdb+profile_6133.npy.gz")
        # path ="D:\\scratch\\hhjs\\dataset\\stupid.npy.gz.npy"
        if not os.path.exists(path):
            if download == True :
                raise NotImplementedError('The download function is not yet implemented') # TODO
            else :
                raise NotImplementedError('The data has not been downloaded, should use download argument') # TODO : Change the exception
        self.noisy = noisy
        self.noise_function = noise_function

        ds = np.load(path)
        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))

    
        ret = np.zeros((ds.shape[0], ds.shape[1], amino_acid_residues + self.num_classes),dtype=np.int32)
        ret[:, :, :amino_acid_residues] = ds[:, :, :amino_acid_residues]
        ret[:, :, amino_acid_residues:] = ds[:, :, amino_acid_residues :amino_acid_residues+ self.num_classes]
    

        D_train, D_test, D_val = split_with_shuffle(ret)

        self.X_train, self.Y_train = self.get_data_labels(D_train)
        self.X_test, self.Y_test = self.get_data_labels(D_test)
        self.X_val, self.Y_val = self.get_data_labels(D_val)


        self.X_train = torch.tensor(self.reshape_data(self.X_train))
        self.X_test = torch.tensor(self.reshape_data(self.X_test))
        self.X_val = torch.tensor(self.reshape_data(self.X_val))

        self.Y_train = torch.tensor(self.reshape_labels(self.Y_train), dtype = torch.int64)
        self.Y_train = torch.argmax(self.Y_train, axis = 1)
        self.Y_test = torch.tensor(self.reshape_labels(self.Y_test), dtype = torch.int64)
        self.Y_test = torch.argmax(self.Y_test, axis = 1)
        self.Y_val = torch.tensor(self.reshape_labels(self.Y_val), dtype = torch.int64)
        

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function)

    def reshape_data(self, X):
        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues))
        print(res.shape)
        for i in range(X.shape[1] - self.cnn_width + 1):
            res[:, i, :, :] = X[:, i:i+self.cnn_width, :]
        res = np.reshape(res, (X.shape[0]*(X.shape[1] - self.cnn_width + 1),amino_acid_residues, self.cnn_width))
      
        return res


    def reshape_labels(self, labels):
        labels = labels[:, int(self.cnn_width/2):-int(self.cnn_width/2), :]
        Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))
        return Y


    def get_data_labels(self, D):
        X = D[:, :, 0:amino_acid_residues]
        Y = D[:, :, amino_acid_residues:amino_acid_residues + self.num_classes]
        return X, Y

##### ENCAPSULATION :


class LoaderProtein():
    def __init__(self, dataset, batch_size_train = 64, batch_size_test = 1024, noisy = False, noise_function=None, root_dir = "/files/"):
        self.root_dir = root_dir

        self.dataset = dataset(root = root_dir, noisy = noisy, noise_function=noise_function)
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader( self.dataset_train,
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                self.dataset_test,
                            batch_size=batch_size_test, shuffle=True
                            )

    def get_category(self):
        return self.dataset.num_classes

    def get_shape(self):
        return self.dataset_train.dataset.shape[1:]

