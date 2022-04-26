import torch
import numpy as np
import scipy

from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented




class UniformDataset(ArtificialDataset):
    def __init__(self, nb_sample_train = 10000, nb_sample_test = 10000, min = -2.0, max = 2.0, dim_input = 2, give_index = False, noise_function = None,  **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.dim_input = dim_input
        self.min = min
        self.max = max
        self.center = 0

        self.nb_sample = nb_sample_test + nb_sample_train
        min_x = np.full((self.dim_input,), min)
        max_x = np.full((self.dim_input,), max)
        # self.X = scipy.stats.uniform(min_x, max_x).rvs((self.nb_sample, self.dim_input))
        # self.X = torch.tensor(self.X, dtype = torch.float32,)
        self.X = (torch.rand((self.nb_sample, self.dim_input),)- 0.5) * (self.min-self.max) + self.center
       
    def impute(self, value,  mask, index = None, dataset_type=None): 
        sampled = (torch.rand(value.shape, device = mask.device) - 0.5) * (self.min-self.max) + self.center
        return sampled


class DiagDataset(UniformDataset):
    def __init__(self, nb_sample_train = 10000, nb_sample_test = 10000, min = -2.0, max = 2.0, give_index = False, noise_function = None, dim_input = 2, **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, min = min, max = max, dim_input = dim_input, give_index = give_index, noise_function = noise_function, **kwargs)

        self.Y = np.where(self.X[:,0]<self.X[:,1], np.ones((self.nb_sample)), np.zeros((self.nb_sample))).astype(np.int64)
        self.data_train = torch.tensor(self.X[:nb_sample_train], dtype= torch.float32)
        self.target_train = torch.tensor(self.Y[:nb_sample_train], dtype = torch.int64)
        
        self.data_test = torch.tensor(self.X[nb_sample_train:], dtype=torch.float32)
        self.target_test = torch.tensor(self.Y[nb_sample_train:], dtype = torch.int64)

        self.optimal_S_train = torch.ones_like(self.data_train)
        self.optimal_S_test = torch.ones_like(self.data_test)
        self.nb_classes = 2

        
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)
    



    def get_true_selection(self, index,  train_dataset = True):
        if not self.give_index :
            raise AttributeError("You need to give the index in the distribution if you want to use true Selection as input of a model")
        if train_dataset :
            true_S = self.optimal_S_train
        else :
            true_S = self.optimal_S_test

        if index.is_cuda :
            true_S = true_S.cuda()
            
        true_S_value = true_S[index]
        return true_S_value


    def get_true_output(self, value, mask=None, index=None, dataset_type = None):
        if mask is None :
            mask = torch.ones_like(value, dtype=torch.float32)
        
        aux_value = mask * value
        output = torch.where(aux_value[:,0]<aux_value[:,1], torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        aux_output = torch.sum((2 - aux_value)/4 * mask, dim = -1)
        
        output = torch.where(torch.all(mask >0.5 , dim = -1), output, aux_output)
        output = torch.where(torch.all(mask <0.5, dim = -1), torch.tensor(0.5, dtype = torch.float32), output)
        
        output = output.unsqueeze(-1)
        output = torch.cat((output, torch.ones_like(output) - output), dim = -1)

        return output  




class ExpProdUniformDataset(UniformDataset):
    def __init__(self, nb_sample_train = 10000, nb_sample_test = 10000, min = -2.0, max = 2.0, dim_input = 2, used_dim = 2, give_index = False, noise_function = None,  **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, min = min, max = max, dim_input = dim_input, give_index = give_index, noise_function = noise_function, **kwargs)

        self.used_dim = used_dim
        self.nb_classes = 2
        fa = torch.exp(torch.prod(self.X[:,:used_dim], axis = 1))
        b_fa = 1/(1+fa)
        sel = torch.zeros_like(self.X)
        sel[:,:used_dim] = 1
        print(b_fa.mean())
        Y = torch.rand(size = b_fa.shape)
        Y = torch.where(Y<b_fa, torch.ones_like(b_fa, dtype = torch.int64), torch.zeros_like(b_fa, dtype = torch.int64))

        self.data_train = self.X[:self.nb_sample_train,:]
        self.data_test = self.X[self.nb_sample_train:,:]
        self.target_train = Y[:self.nb_sample_train,]
        self.target_test = Y[self.nb_sample_train:,]
        self.optimal_S_train = sel[:nb_sample_train,:]
        self.optimal_S_test = sel[nb_sample_train:,:]
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)
    

class ExpSquaredSumUniformDataset(UniformDataset):
    def __init__(self, nb_sample_train = 10000, nb_sample_test = 10000, min = -2.0, max = 2.0, dim_input = 2, used_dim = 2, give_index = False, noise_function = None,  **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, min = min, max = max, dim_input = dim_input, give_index = give_index, noise_function = noise_function, **kwargs)
        self.used_dim = used_dim
        self.nb_classes = 2
        fa = torch.exp(torch.sum(self.X[:,:used_dim]**2 , axis = 1) - 4)
        b_fa = 1/(1+fa)
        sel = torch.zeros_like(self.X)
        sel[:,:used_dim] = 1

        self.Y = torch.rand(size = b_fa.shape,)
        self.Y = torch.where(self.Y<b_fa, torch.ones_like(b_fa, dtype = torch.int64), torch.zeros_like(b_fa, dtype = torch.int64))

        self.data_train = self.X[:self.nb_sample_train,:]
        self.data_test = self.X[self.nb_sample_train:,:]
        self.target_train = self.Y[:self.nb_sample_train,]
        self.target_test = self.Y[self.nb_sample_train:,]
        self.optimal_S_train = sel[:nb_sample_train,:]
        self.optimal_S_test = sel[nb_sample_train:,:]
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)


class ExpSquaredSumUniformDatasetV2(UniformDataset):
    def __init__(self, nb_sample_train = 10000, nb_sample_test = 10000, min = -2.0, max = 2.0, dim_input = 2, used_dim = 2, give_index = False, noise_function = None,  **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, min = min, max = max, dim_input = dim_input, give_index = give_index, noise_function = noise_function, **kwargs)
        self.used_dim = used_dim
        self.nb_classes = 2
        fa = torch.exp(torch.sum(self.X[:,:used_dim]**2 , axis = 1) - used_dim)
        b_fa = 1/(1+fa)
        sel = torch.zeros_like(self.X)
        sel[:,:used_dim] = 1

        self.Y = torch.rand(size = b_fa.shape,)
        self.Y = torch.where(self.Y<b_fa, torch.ones_like(b_fa, dtype = torch.int64), torch.zeros_like(b_fa, dtype = torch.int64))

        self.data_train = self.X[:self.nb_sample_train,:]
        self.data_test = self.X[self.nb_sample_train:,:]
        self.target_train = self.Y[:self.nb_sample_train,]
        self.target_test = self.Y[self.nb_sample_train:,]
        self.optimal_S_train = sel[:nb_sample_train,:]
        self.optimal_S_test = sel[nb_sample_train:,:]
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)
    