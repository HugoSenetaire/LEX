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


torch.pi = torch.tensor(3.1415)
np.random.seed(0)
torch.manual_seed(0)


def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised


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

        
# from numbers import Number

# import torch
# from torch.distributions import constraints
# from torch.distributions.distribution import Distribution
# from torch.distributions.utils import broadcast_all

# class CircleDatasetDitribution(Distribution):
#     # TODO allow (loc,scale) parameterization to allow independent constraints.
#     arg_constraints = {'mask': constraints.dependent(is_discrete=False, event_dim=0),
#                        'value': constraints.dependent(is_discrete=False, event_dim=0)}
#     has_rsample = False
    
#     def __init__(self, mask, value, ratio):
#         self.ratio = ratio
#         self.mask, self.value = broadcast_all(mask, value)

#         if isinstance(mask, Number) and isinstance(value, Number):
#             batch_shape = torch.Size()
#         else:
#             batch_shape = self.low.size()
#         super(CircleDatasetDitribution, self).__init__(batch_shape, validate_args=validate_args)

#         if self._validate_args :
#             raise ValueError("Add a method here")

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(CircleDatasetDitribution, _instance)
#         batch_shape = torch.Size(batch_shape)
#         new.mask = self.mask.expand(batch_shape)
#         new.value = self.value.expand(batch_shape)
#         super(CircleDatasetDitribution, new).__init__(batch_shape, validate_args=False)
#         new._validate_args = self._validate_args
#         return new

#     def sample(self, sample_shape = torch.size()):
        
#         x_1_missing = torch.where(self.mask[:,0] == 0, True, False)
#         x_2_missing = torch.where(self.mask[:,1] == 0, True, False)


#     def log_prob(self, value):

    


class CircleDataset():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
        self.factor = factor
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train)
        self.data_test = torch.tensor(self.data_test)
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)


    def impute_result(self, mask, value):
        mask_aux = mask.detach()
        value_aux = value.detach()
        
        ratio = torch.tensor(self.factor)
        batch_size = value.shape[0]

        x_1_missing = torch.where(mask_aux[:,0] == 0, True, False)
        x_2_missing = torch.where(mask_aux[:,1] == 0, True, False)

        both_missing = torch.where(x_1_missing, x_2_missing, False)
        none_missing = torch.where(x_1_missing, False , True)
        none_missing = torch.where(x_2_missing, False, none_missing).unsqueeze(-1).expand(-1,2)
        ratio_vector = torch.empty(0).new_full(value.shape, ratio).cuda()

        x_1_missing = torch.where(both_missing, False, x_1_missing)
        x_1_missing_sup_r1 = torch.where(torch.abs(value[:,1])>ratio_vector[:,1], x_1_missing, False)
        
        x_2_missing = torch.where(both_missing, False, x_2_missing)
        x_2_missing_sup_r2 = torch.where(torch.abs(value[:,0])>ratio_vector[:,1], x_2_missing, False)

        both_missing = both_missing.unsqueeze(-1).expand(-1,2)

        complete_output = value_aux.clone()

        # First both_missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size) * 0.5).unsqueeze(-1).expand(-1,2).cuda()
        uniform = 2*torch.pi* torch.rand(batch_size).cuda()
        new_output = (bernoulli + (1-bernoulli) * ratio) * torch.cat([torch.cos(uniform).unsqueeze(-1), torch.sin(uniform).unsqueeze(-1)],-1)
        complete_output = torch.where(both_missing, new_output, complete_output)

        # x_1 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_sup_r1 = (2*bernoulli-1) * torch.sqrt(1 - value[:,1]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_inf_r1 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,1])**2,torch.zeros(batch_size).cuda()))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()

        complete_output[:,0] = torch.where(x_1_missing,
                            bernoulli * x_1_missing_new_value_sup_r1 + (1-bernoulli) * x_1_missing_new_value_inf_r1,
                            complete_output[:,0])

        complete_output[:,0] = torch.where(x_1_missing_sup_r1, x_1_missing_new_value_sup_r1,
                            complete_output[:,0]) 


        # x_2 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_2_missing_new_value_sup_r2 = (2*bernoulli-1) * torch.sqrt(1 - value[:,0]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_2_missing_new_value_inf_r2 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,0])**2,torch.zeros(batch_size).cuda()))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()

        complete_output[:,1] = torch.where(x_2_missing,
                            bernoulli * x_2_missing_new_value_sup_r2 + (1-bernoulli) * x_2_missing_new_value_inf_r2,
                            complete_output[:,1])

        complete_output[:,1] = torch.where(x_2_missing_sup_r2, x_2_missing_new_value_sup_r2,
                            complete_output[:,1])


        # None missing 
        complete_output = torch.where(none_missing, value, complete_output)


        return complete_output

class CircleDatasetAndNoise():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.noise = np.random.normal(loc=0.0, scale=1.0, size=(total_samples, 1))
        self.data = np.concatenate([self.data, self.noise], axis = 1)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train)
        self.data_test = torch.tensor(self.data_test)
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)


    def impute_result(mask, value, ratio = 0.6):
        mask_aux = mask.detach()
        value_aux = value.detach()
        ratio = torch.tensor(ratio)
        batch_size = value.shape[0]

        x_1_missing = torch.where(mask_aux[:,0] == 0, True, False)
        x_2_missing = torch.where(mask_aux[:,1] == 0, True, False)

        both_missing = torch.where(x_1_missing, x_2_missing, False)
        none_missing = torch.where(x_1_missing, False , True)
        none_missing = torch.where(x_2_missing, False, none_missing).unsqueeze(-1).expand(-1,2)
        ratio_vector = torch.empty(0).new_full(value.shape, ratio)

        x_1_missing = torch.where(both_missing, False, x_1_missing)
        x_1_missing_sup_r1 = torch.where(torch.abs(value[:,1])>ratio_vector[:,1], x_1_missing, False)
        
        x_2_missing = torch.where(both_missing, False, x_2_missing)
        x_2_missing_sup_r2 = torch.where(torch.abs(value[:,0])>ratio_vector[:,1], x_2_missing, False)

        both_missing = both_missing.unsqueeze(-1).expand(-1,2)

        complete_output = value_aux.clone()

        # First both_missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size) * 0.5).unsqueeze(-1).expand(-1,2)
        uniform = 2*torch.pi* torch.rand(batch_size)
        new_output = (bernoulli + (1-bernoulli) * ratio) * torch.cat([torch.cos(uniform).unsqueeze(-1), torch.sin(uniform).unsqueeze(-1)],-1)
        complete_output = torch.where(both_missing, new_output, complete_output)

        # x_1 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)
        x_1_missing_new_value_sup_r1 = (2*bernoulli-1) * torch.sqrt(1 - value[:,1]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)
        x_1_missing_new_value_inf_r1 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,1])**2,torch.zeros(batch_size)))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)

        complete_output[:,0] = torch.where(x_1_missing,
                            bernoulli * x_1_missing_new_value_sup_r1 + (1-bernoulli) * x_1_missing_new_value_inf_r1,
                            complete_output[:,0])

        complete_output[:,0] = torch.where(x_1_missing_sup_r1, x_1_missing_new_value_sup_r1,
                            complete_output[:,0]) 


        # x_2 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)
        x_2_missing_new_value_sup_r2 = (2*bernoulli-1) * torch.sqrt(1 - value[:,0]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)
        x_2_missing_new_value_inf_r2 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,0])**2,torch.zeros(batch_size)))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5)

        complete_output[:,1] = torch.where(x_2_missing,
                            bernoulli * x_2_missing_new_value_sup_r2 + (1-bernoulli) * x_2_missing_new_value_inf_r2,
                            complete_output[:,1])

        complete_output[:,1] = torch.where(x_2_missing_sup_r2, x_2_missing_new_value_sup_r2,
                            complete_output[:,1])


        # None missing 
        complete_output = torch.where(none_missing, value, complete_output)


        return complete_output


class CircleDatasetNotCentered():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, shift = [2,2], factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)
        self.noise = np.random.normal(loc=0.0, scale=1.0, size=(total_samples, 1))
        self.data = np.concatenate([self.data, self.noise], axis = 1)


        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)



        self.data_train = torch.tensor(self.data_train + np.array(shift))
        self.data_test = torch.tensor(self.data_test + np.array(shift))
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)


class CircleDatasetNotCenteredAndNoise():
    def __init__(self, n_samples_train = 40000, n_samples_test=10000, noise = False, shift = [2,2], factor =.6, noisy = False, noise_function = None):

        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise
     
        total_samples = self.n_samples_train + self.n_samples_test
        test_size = self.n_samples_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train + np.array(shift))
        self.data_test = torch.tensor(self.data_test + np.array(shift))
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)

##### ENCAPSULATION :

class LoaderArtificial():
    def __init__(self,dataset, batch_size_train = 1024, batch_size_test=1000, n_samples_train = 100000, n_samples_test=10000, noisy = False):

        self.dataset = dataset(n_samples_train = n_samples_train, n_samples_test = n_samples_test, noisy = noisy)
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader( self.dataset_train,
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                self.dataset_test,
                            batch_size=batch_size_test, shuffle=False
                            )

    def get_category(self):
        return 2

    def get_shape(self):
        return (2)

