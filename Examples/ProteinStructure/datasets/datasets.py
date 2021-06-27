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
amino_acid_residues = 21



class TensorDatasetAugmented(TensorDataset):
    def __init__(self, x, y, noisy = False, noise_function = None):
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

def split_from_paper(Dataset, seed=0):
    # Dataset subdivision following dataset readme and paper
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(Dataset)
    Train = Dataset[0:5600, :, :]
    Test = Dataset[5600:5877, :, :]
    Validation = Dataset[5877:, :, :]
    return Train, Test, Validation

def split_with_shuffle(Dataset, seed=0):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0]*0.8)
    test_val_split = int(Dataset.shape[0]*0.1)
    Train = Dataset[0:train_split, :, :]
    Test = Dataset[train_split:train_split+test_val_split, :, :]
    Validation = Dataset[train_split+test_val_split:, :, :]
    return Train, Test, Validation


class cullpdb_6133_8classes_asinpaper():
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
        if not os.path.exists(path):
            if download == True :
                raise NotImplementedError('The download function is not yet implemented') # TODO
            else :
                raise NotImplementedError('The data has not been downloaded, should use download argument') # TODO : Change the exception
        self.noisy = noisy
        self.noise_function = noise_function
        
        ds = np.load(path)

        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
        # ret = np.zeros((ds.shape[0], ds.shape[1], amino_acid_residues + self.num_classes),dtype=np.int32)
        # ret[:, :, 0:amino_acid_residues] = ds[:, :, 35:56]
        # print(np.any(ds[:,:, 35:56]>0))
        ret = np.concatenate([ds[:,:, 35:56], ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + self.num_classes]],axis = -1)
        # ret[:, :, amino_acid_residues:] = ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + self.num_classes]
        # print(np.any(ret[:, :, 0:amino_acid_residues]>0))


       

        D_train, D_test, D_val = split_from_paper(ret)

        self.X_train, self.Y_train = self.get_data_labels(D_train)
        self.X_test, self.Y_test = self.get_data_labels(D_test)
        self.X_val, self.Y_val = self.get_data_labels(D_val)


        print("Shape before reshape Xtrain", np.shape(self.X_train))
        print("Shape before reshape Ytrain", np.shape(self.Y_train))
        

        self.X_train = torch.tensor(self.reshape_data(self.X_train))
        self.X_test = torch.tensor(self.reshape_data(self.X_test))
        self.X_val = torch.tensor(self.reshape_data(self.X_val))

        self.Y_train = torch.tensor(self.reshape_labels(self.Y_train), dtype = torch.int64)

        print("Shape after reshape Xtrain", np.shape(self.X_train))
        print("Shape after reshape Ytrain", np.shape(self.Y_train))
        self.Y_train = torch.argmax(self.Y_train, axis = 1)
        self.Y_test = torch.tensor(self.reshape_labels(self.Y_test), dtype = torch.int64)
        self.Y_test = torch.argmax(self.Y_test, axis = 1)
        self.Y_val = torch.tensor(self.reshape_labels(self.Y_val), dtype = torch.int64)

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function)

  def reshape_data(self, X):
        padding = np.zeros((X.shape[0], X.shape[2], int(self.cnn_width/2)))
        X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))

        X = np.swapaxes(X, 1, 2)
        print(np.any(X>0))
        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues))
        for i in range(X.shape[1] - self.cnn_width + 1):
            res[:, i, :, :] = X[:, i:i+self.cnn_width, :]
            # print(X[:, i:i+self.cnn_width, :])
            # if np.any(X[:, i:i+self.cnn_width, :]>0):
              # print(X[:, i:i+self.cnn_width, :])
            # break
        res = np.reshape(res, (X.shape[0]*(X.shape[1] - self.cnn_width + 1),amino_acid_residues, self.cnn_width))

        # print(res[0])
        res = res[np.count_nonzero(res, axis=(1,2))>(int(self.cnn_width/2)*amino_acid_residues), :, :]
        return res


  def reshape_labels(self, labels):
        Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))
        Y = Y[~np.all(Y == 0, axis=1)]
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
        if not os.path.exists(path):
            if download == True :
                raise NotImplementedError('The download function is not yet implemented') # TODO
            else :
                raise NotImplementedError('The data has not been downloaded, should use download argument') # TODO : Change the exception
        self.noisy = noisy
        self.noise_function = noise_function
        
        ds = np.load(path)
        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))


        ret = np.concatenate([ds[:,:, :amino_acid_residues+1], ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + self.num_classes]],axis = -1) # Actually the part no seq in Y is not considered.

        D_train, D_test, D_val = split_from_paper(ret)

        self.X_train, self.Y_train = self.get_data_labels(D_train)
        self.X_test, self.Y_test = self.get_data_labels(D_test)
        self.X_val, self.Y_val = self.get_data_labels(D_val)


        self.X_train, self.Y_train = self.reshape_data(self.X_train, self.Y_train)
        self.X_test, self.Y_test = self.reshape_data(self.X_test, self.Y_test)
        self.X_val, self.Y_val = self.reshape_data(self.X_val, self.Y_val)

        self.X_train = torch.tensor(self.X_train)
        self.Y_train = torch.tensor(self.Y_train, dtype= torch.int64)
        self.X_test = torch.tensor(self.X_test)
        self.Y_test = torch.tensor(self.Y_test, dtype= torch.int64)
        self.X_val = torch.tensor(self.X_val)
        self.Y_val = torch.tensor(self.Y_val, dtype= torch.int64)

        # self.Y_train = torch.tensor(self.reshape_labels(self.Y_train), dtype = torch.int64)
        self.Y_train = torch.argmax(self.Y_train, axis = 1)
        # self.Y_test = torch.tensor(self.reshape_labels(self.Y_test), dtype = torch.int64)
        self.Y_test = torch.argmax(self.Y_test, axis = 1)
        # self.Y_val = torch.tensor(self.reshape_labels(self.Y_val), dtype = torch.int64)
        

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function)

    def reshape_data(self, X, labels):

        padding = np.concatenate([np.zeros((X.shape[0],X.shape[2]-1,int(self.cnn_width/2))), np.ones((X.shape[0],1, int(self.cnn_width/2)))], axis=1)
        X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
        X = np.swapaxes(X, 1, 2)

        Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues+1))
        for i in range(X.shape[1] - self.cnn_width + 1):
            aux = X[:, i:i+self.cnn_width, :]
            res[:, i, :, :] = X[:, i:i+self.cnn_width, :]
            aux = res[:,i,:,:]
         
      
        res = np.reshape(res, (np.shape(res)[0]*np.shape(res)[1], self.cnn_width, amino_acid_residues+1))

        res = np.transpose(res, (0,2,1))
        Y = Y[~np.any(np.argmax(res, axis=-2)==21, axis =-1)]
        res = res[~np.any(np.argmax(res, axis=-2)==21, axis=-1)]
        res = res[:,:amino_acid_residues,:]
        auxres1 = np.sum(res,axis=-2)
        print("problem with 0", len(np.where(auxres1 !=1)[0]))
        auxres2 = np.argmax(res, axis=-2)
        print("Problem with no seq", len(np.where(auxres2==21)[0]))
        
        return res, Y



    def get_data_labels(self, D):
        X = D[:, :, 0:amino_acid_residues+1]
        Y = D[:, :, amino_acid_residues+1:amino_acid_residues+1 + self.num_classes]
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

