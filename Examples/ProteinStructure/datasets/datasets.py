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
    def __init__(self, x, y, noisy = False, noise_function = None, give_index = False):
        super().__init__(x,y)
        self.noisy = noisy
        self.noise_function = noise_function
        self.give_index = give_index


    def __getitem__(self, idx):
        if not self.noisy :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx]
            if self.give_index :
                index = torch.tensor(idx).type(torch.int64)
                return input_tensor, target, index
            return input_tensor, target
        else :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx].type(torch.float32)
            input_tensor = input_tensor.numpy()
            target = target.numpy()
            input_tensor = torch.tensor(self.noise_function(input_tensor)).type(torch.float32)
            if self.give_index :
                index = torch.tensor(idx).type(int64)
                return input_tensor, target, index

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


def change_sequence_profile_local(ds, dim1=1, dim2=2):
    ds[:,:,dim1], ds[:,:,dim2] = ds[:,:,dim2], ds[:,:,dim1]
    return ds

def change_sequence_profile(ds):
    list_to_change = [(2,3), (4,5), (6,7), (9,10), (12,13), (14,15), (17,18), (19,20)]
    for dim1, dim2 in list_to_change:
        ds = change_sequence_profile_local(ds, dim1=dim1, dim2=dim2)

    return ds


class cullpdb_6133_8classes_asinpaper():
  def __init__(self,
            root: str,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
            cnn_width = 19,
            give_index = False,
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
        self.give_index = give_index
        
        ds = np.load(path)


        # path_aux = "D:\\scratch\\hhjs\\dataset\\cullpdb_reduced.npy"
        # ds = np.load(path_aux)

        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
        ret = np.concatenate([ds[:,:, 35:56], ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + self.num_classes]],axis = -1)



       

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

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function, give_index =self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function, give_index = self.give_index)

  def reshape_data(self, X):
        padding = np.zeros((X.shape[0], X.shape[2], int(self.cnn_width/2)))
        X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))

        X = np.swapaxes(X, 1, 2)
        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues))
        for i in range(X.shape[1] - self.cnn_width + 1):
            res[:, i, :, :] = X[:, i:i+self.cnn_width, :]

        res = np.reshape(res, (X.shape[0]*(X.shape[1] - self.cnn_width + 1),amino_acid_residues, self.cnn_width))

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
            give_index = False,
            sampling_imputation = True,
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
        self.give_index = give_index
        self.sampling_imputation = sampling_imputation

                
        # path_aux = "D:\\scratch\\hhjs\\dataset\\cullpdb_reduced.npy"
        # path_aux = "D:\\scratch\\hhjs\\dataset\\cullpdb_reduced_shuffle.npy"
        ds = np.load(path)
        # ds = np.load(path_aux)
        ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))


        ret = np.concatenate([ds[:,:, :amino_acid_residues+1], ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + self.num_classes], ds[:,:, 35:56]],axis = -1) # Actually the part no seq in Y is not considered.

        D_train, D_test, D_val = split_with_shuffle(ret)

        self.X_train, self.Y_train, self.X_aux_train = self.get_data_labels(D_train)
        self.X_aux_train = change_sequence_profile(self.X_aux_train)
        self.X_test, self.Y_test, self.X_aux_test = self.get_data_labels(D_test)
        self.X_aux_test = change_sequence_profile(self.X_aux_test)
        self.X_val, self.Y_val, self.X_aux_val = self.get_data_labels(D_val)
        self.X_aux_val = change_sequence_profile(self.X_aux_val)



        self.X_train, self.Y_train, self.X_aux_train = self.reshape_data(self.X_train, self.Y_train, self.X_aux_train)
        self.X_test, self.Y_test, self.X_aux_test = self.reshape_data(self.X_test, self.Y_test, self.X_aux_test)
        self.X_val, self.Y_val, self.X_aux_val = self.reshape_data(self.X_val, self.Y_val, self.X_aux_val)

        self.X_train = torch.tensor(self.X_train)
        self.Y_train = torch.tensor(self.Y_train, dtype= torch.int64)
        self.X_aux_train = torch.tensor(self.X_aux_train)

        self.X_test = torch.tensor(self.X_test)
        self.Y_test = torch.tensor(self.Y_test, dtype= torch.int64)
        self.X_aux_test = torch.tensor(self.X_aux_test)

        self.X_val = torch.tensor(self.X_val)
        self.Y_val = torch.tensor(self.Y_val, dtype= torch.int64)

        # self.Y_train = torch.tensor(self.reshape_labels(self.Y_train), dtype = torch.int64)
        self.Y_train = torch.argmax(self.Y_train, axis = 1)
        # self.Y_test = torch.tensor(self.reshape_labels(self.Y_test), dtype = torch.int64)
        self.Y_test = torch.argmax(self.Y_test, axis = 1)
        # self.Y_val = torch.tensor(self.reshape_labels(self.Y_val), dtype = torch.int64)
        

        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, noisy = noisy, noise_function = noise_function, give_index =self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, noisy= noisy, noise_function = noise_function, give_index = self.give_index)

        if self.give_index:
            self.dataset_imputation_train = TensorDatasetAugmented(self.X_aux_train, self.Y_train, noisy = noisy, noise_function = noise_function, give_index =False)
            self.dataset_imputation_test = TensorDatasetAugmented(self.X_aux_test, self.Y_test, noisy= noisy, noise_function = noise_function, give_index = False)


    def reshape_data(self, X, labels, X_aux):
        padding = np.concatenate([np.zeros((X.shape[0],X.shape[2]-1,int(self.cnn_width/2))), np.ones((X.shape[0],1, int(self.cnn_width/2)))], axis=1)
        X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
        X = np.swapaxes(X, 1, 2)

        padding = np.zeros((X_aux.shape[0],X_aux.shape[2],int(self.cnn_width/2)))
        X_aux = np.dstack((padding, np.swapaxes(X_aux, 1, 2), padding))
        X_aux = np.swapaxes(X_aux, 1, 2)
        Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))




        res = np.zeros((X.shape[0], X.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues+1))
        res_aux = np.zeros((X_aux.shape[0], X_aux.shape[1] - self.cnn_width + 1, self.cnn_width, amino_acid_residues))


        for i in range(X.shape[1] - self.cnn_width + 1):            
                res[:, i, :, :] = X[:, i:i+self.cnn_width, :]
                
                res_aux[:,i,:,:] = X_aux[:, i:i+self.cnn_width, :]
                

        res = np.reshape(res, (np.shape(res)[0]*np.shape(res)[1], self.cnn_width, amino_acid_residues+1))
        res_aux = np.reshape(res_aux, (np.shape(res_aux)[0]*np.shape(res_aux)[1], self.cnn_width, amino_acid_residues))
        res = res[:,:,:-1]


        index_delete_res_aux = np.reshape(np.any(np.sum(res_aux, axis=-1)== 0, axis=-1),(-1,1))
        index_delete_res = np.reshape(np.any(np.sum(res, axis=-1)==0,axis=-1),(-1,1))
        index_delete_Y = np.reshape(np.sum(Y, axis=-1)==0,(-1,1))

        index_to_delete = np.concatenate([
                            index_delete_res,
                            index_delete_res_aux,
                            index_delete_Y,
                            ],
                            axis=-1)

        index_to_delete = np.any(index_to_delete,axis=-1)


        Y = Y[            ~index_to_delete]
        res_aux = res_aux[~index_to_delete]
        res = res[        ~index_to_delete]

        
        res = np.transpose(res, (0,2,1))
        res_aux = np.transpose(res_aux, (0,2,1))


        print(res.shape)
        auxres1 = np.sum(res,axis=-2)
        print("problem with 0", len(np.where(auxres1 == 0)[0]))
        auxres_aux1 = np.sum(res_aux,axis=-2)
        print("problem with 0 aux", len(np.where(auxres_aux1 ==0)[0]))
        auxY1 = np.sum(Y,axis=-1)
        print("problem with 0 Y", len(np.where(auxY1 ==0)[0]))

        return res, Y, res_aux


    def impute_result(self, mask, value, index, dataset_type = "Train"):
        index = index.flatten(0)
        if dataset_type == "Train":
            output, _ = self.dataset_imputation_train.__getitem__(index)
            original_value,_,_ = self.dataset_train.__getitem__(index)
        else :
            output, _ = self.dataset_imputation_test.__getitem__(index)

        if self.sampling_imputation :
            dist = torch.distributions.categorical.Categorical(probs = output.transpose(1,2))
            imputed_value = torch.nn.functional.one_hot(dist.sample(), num_classes=amino_acid_residues).transpose(-1,-2)
        else :
            imputed_value = output

        if value.is_cuda:
            imputed_value = imputed_value.cuda() 
        

        new_data = imputed_value *  (1-mask) + value * mask 

        return new_data 

    def get_data_labels(self, D):
        X = D[:, :, 0:amino_acid_residues+1]
        Y = D[:, :, amino_acid_residues+1:amino_acid_residues+1 + self.num_classes]
        X_aux = D[:,:, amino_acid_residues+1 + self.num_classes:]
        return X, Y, X_aux
##### ENCAPSULATION :


class LoaderProtein():
    def __init__(self, dataset, batch_size_train = 64, batch_size_test = 1024, noisy = False, noise_function=None, root_dir = "/files/"):
        self.root_dir = root_dir

        self.dataset = dataset(root = root_dir, noisy = noisy, noise_function=noise_function)
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
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

