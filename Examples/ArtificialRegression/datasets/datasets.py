import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader, dataset
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn


from sklearn import cluster, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy


torch.pi = torch.tensor(3.1415)
np.random.seed(0)
torch.manual_seed(0)


def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised



def bit_count(bin_subset):
    ''' Cardinality of the set corresponding to bin_subset. '''
    c = 0
    while bin_subset:
        bin_subset &= bin_subset - 1
        c += 1
    return c

def gen_multivariate(n, d, max_sel_dim = 2, sigma = 0.25, prob_simplify = 0.2, mix_shape=True, exact_sel_dim = False):
    '''
    Generates n shapes in d dimensions
    
    * max_sel_dim: maximum selection cardinality
    * sigma: min dist between points -> used to tune the Gaussian variances
    * prob_simplify: probability to delete some generated centroids to bring diversity
      in the shapes. For prob = 0., we generate hyper-cubes with binary labels (as an
      hypercube defines a bipartite graph, it's possible to do that.) 
    * mix_shape: a debugging parameter, use False to see all shapes nicely aligned
      along a \vec{(1, 1, ...)} line.
    '''
    assert max_sel_dim <= d
    X = []
    Y = []
    S = []
    n_points = 0
    shapes = []
    to_delete = []

    # 1 -- generate abstract problems
    used = [0 for i in range(d)]
    for i in range(n):
        # select a ground-truth solution for the shape
        if exact_sel_dim :
            sel_dim = max_sel_dim
        else :
            sel_dim = np.random.randint(1, max_sel_dim + 1)
        sel = sorted(list(np.random.permutation(d)[:sel_dim]))

        # allocate some coordinates that do not collide with previous ones
        x_base = []
        for k in range(d):
            x_base.append(used[k])
            used[k] += 1
        x_separated = list(x_base)  # copy
        for k in sel:
            x_separated[k] = used[k]
            used[k] += 1

        # add a binary hypercube in sel_dim dimensions
        shapes.append([])
        for j in range(2 ** sel_dim):
            x = list(x_base)
            for i_k, k in enumerate(sel):
                if 1 << i_k & j > 0:
                    x[k] = x_separated[k]  # move point along dim k
            X.append(x)
            if bit_count(j) % 2 == 0:
                Y.append(1.)
            else:
                Y.append(0.)
            S.append(list(sel))  # don't forget to copy! else the removal fails
            shapes[-1].append(n_points)
            n_points += 1

        # delete hypercube points to create more complex selections
        for j in range(2 ** sel_dim):
            if np.random.random() < prob_simplify and bit_count(j) > 1:
                if shapes[-1][0] + j not in to_delete:
                    to_delete.append(shapes[-1][0] + j)
                for l in range(sel_dim):  # frees dependences for neighbors
                    neighbor = j ^ (1 << l)  # flip one dim to find neighbor
                    neighbor_id = shapes[-1][neighbor]
                    S[neighbor_id].remove(sel[l])
                    if len(S[neighbor_id]) == 0 and (neighbor_id not in to_delete):
                        to_delete.append(neighbor_id)  # don't keep isolated point

    # effectively delete points from all lists
    for point_to_del in sorted(to_delete)[::-1]:
        try:
            X.pop(point_to_del)
            Y.pop(point_to_del)
            S.pop(point_to_del)
        except:
            raise ValueError(point_to_del, sorted(to_delete))
    to_del_ind = 0
    for shape in shapes:  # remove from shapes
        while to_del_ind < len(to_delete) and to_delete[to_del_ind] in shape:
            shape.remove(to_delete[to_del_ind])
            to_del_ind += 1
        if to_del_ind >= len(to_delete):
            break
    acc_shape = 0
    for i, shape in enumerate(shapes):  # reorder points from 1 to n.
        l_shape = len(shape)
        shapes[i] = list(range(acc_shape, acc_shape + l_shape))
        acc_shape += l_shape

    # 2 -- translate into real points spaced by sigma
    real_coord = [np.linspace(0, sigma * (k_used - 1), k_used) for k_used in used]
    for k in range(d):
        real_coord[k] = real_coord[k] - np.mean(real_coord[k])
        if mix_shape:
            real_coord[k] = np.random.permutation(real_coord[k])
    X_real = np.zeros((len(X), d))
    for i, x in enumerate(X):
        for k in range(d):
            X_real[i,k] = real_coord[k][X[i][k]]
    Y_real = np.array(Y, dtype='float32')


    return X_real, Y_real, S, shapes

def generate_distribution_local(centroids_X, centroids_Y, new_S, sigma, nb_samples = 20):
    nb_point, nb_dim = centroids_X.shape

    augmented_X = centroids_X.unsqueeze(1).expand(-1, nb_samples, -1).flatten(0,1)
    new_S_reshaped = new_S.unsqueeze(1).expand(-1,nb_samples,-1).flatten(0,1)

    Y = centroids_Y.unsqueeze(-1).expand(-1, nb_samples).flatten(0,1)
    X = augmented_X + torch.normal(torch.zeros_like(augmented_X), std = sigma)

    # X = X.type(torch.float32)

    
    return X,Y, new_S_reshaped

def generate_distribution(centroids_X, centroids_Y, new_S, sigma, nb_sample_train = 20, nb_sample_test = 20):
  X_train, Y_train, new_S_train = generate_distribution_local(centroids_X, centroids_Y, new_S, sigma, nb_sample_train)
  X_test, Y_test, new_S_test = generate_distribution_local(centroids_X, centroids_Y, new_S, sigma, nb_sample_test) 

  return X_train, Y_train, new_S_train, X_test, Y_test, new_S_test


def redraw_dependency(S, nb_dim):
  nb_shape = len(S)
  new_S = torch.zeros((nb_shape, nb_dim))
  for k in range(len(S)):
    new_S[k, S[k]] = torch.ones(len(S[k]))

  return new_S
 



class TensorDatasetAugmented(TensorDataset):
    def __init__(self, x, y, noisy = False, noise_function = None, give_index = False):
        super().__init__(x,y)
        self.noisy = noisy
        self.noise_function = noise_function
        self.give_index = give_index 


    def __getitem__(self, idx):
        if not self.noisy :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx]
        else :
            input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx].type(torch.float32)
            
            input_tensor = input_tensor.numpy()
            target = target.numpy()
            input_tensor = torch.tensor(self.noise_function(input_tensor)).type(torch.float32)



        if self.give_index :
            return input_tensor, target, idx
        else :
            return input_tensor, target

class GeneratingTensorDataset(Dataset):
    def __init__(self, function, len, noisy=False, noise_function=None, give_index = False):
        super().__init__()
        self.function = function
        self.len = len
        self.give_index = give_index


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data, y = self.function()
        input_tensor, target = torch.tensor(data.type(torch.float32)), torch.tensor(y.type(torch.int64))
        



        if self.give_index :
            return input_tensor, target, idx
        else :
            return input_tensor, target



class CircleDataset():
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, noise = False, factor =.6, noisy = False, noise_function = None):

        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.noise = noise
        self.factor = factor
     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

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


    def impute_result(self, mask, value, index = None , dataset_type= None):
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
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, noise = False, factor =.6, noisy = False, noise_function = None):

        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.noise = noise
        self.factor = factor
     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

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


    def impute_result(self, mask, value, index = None, dataset_type=None):
        mask_aux = mask.detach()
        value_aux = value.detach()
        ratio = torch.tensor(self.factor).cuda()
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

        complete_output = copy.deepcopy(value_aux)

        # First both_missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size) * 0.5).unsqueeze(-1).expand(-1,2).cuda()
        uniform = 2*torch.pi* torch.rand(batch_size).cuda()
        new_output = (bernoulli + (1-bernoulli) * ratio) * torch.cat([torch.cos(uniform).unsqueeze(-1), torch.sin(uniform).unsqueeze(-1)],-1)
        complete_output[:,:2] = torch.where(both_missing, new_output, complete_output[:,:2])

        # x_1 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_sup_r1 = (2*bernoulli-1) * torch.sqrt(1 - value[:,1]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_inf_r1 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,1])**2, torch.zeros(batch_size).cuda()))

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
        complete_output[:,:2] = torch.where(none_missing, value[:,:2], complete_output[:,:2])
        complete_output[:,2] =  torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=(batch_size))).cuda()

        return complete_output


class CircleDatasetNotCentered():
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, noise = False, shift = [2,2], factor =.6, noisy = False, noise_function = None):

        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.noise = noise
     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

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
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, noise = False, shift = [2,2], factor =.6, noisy = False, noise_function = None):

        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.noise = noise
     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

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

class LinearSeparableDataset():
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, noise = False, shift = [2,2], factor =.1, noisy = False, noise_function = None):
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.noise = noise
     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

        self.data, self.targets = datasets.make_blobs(n_samples=total_samples,centers=2, cluster_std=.1)
        self.data -= np.mean(self.data)
        # self.data /= np.linalg.norm(self.data)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train + np.array(shift))
        self.data_test = torch.tensor(self.data_test + np.array(shift))
        self.targets_train = torch.tensor(self.targets_train, dtype = torch.int64)
        self.targets_test = torch.tensor(self.targets_test, dtype = torch.int64)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noisy = noisy)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noisy= noisy)



class ArtificialDataset(Dataset):
    def __init__(self, nb_sample_train = 20, nb_sample_test = 20, give_index = False, batch_size_train = None,
                 noisy = None, use_cuda = False,) -> None:
        super().__init__()
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.batch_size_train = batch_size_train
        self.noisy = noisy
        self.use_cuda = use_cuda

    def get_true_output(self, value, mask = None, index=None, dataset_type = None):
        raise NotImplementedError("Using abstract class ArtificialDataset")

    def impute(self, value, mask=None, index=None, dataset_type = None):
        raise NotImplementedError("Using abstract class ArtificialDataset")

    def calculate_true_selection_variation(self, X, normalize = False, classifier = None, nb_imputation = 100,):
        """
        Calculate the true selection for the given data X using the true definition of the dataset. If classifier is None, uses the true definition of output from the dataset.
        """
        output_S = np.zeros(X.shape)

        if classifier is None :
            classifier = lambda x: self.get_true_output(x)
        
        nb_batch = X.shape[0] // self.batch_size_train + 1

        for k in range(nb_batch) :
            if k != nb_batch - 1 :
                X_batch = X[k*self.batch_size_train:(k+1)*self.batch_size_train]
            else :
                X_batch = X[k*self.batch_size_train:]

        
            if k*self.batch_size_train == len(X):
  
                continue

            batch_size = X_batch.shape[0]

            X_batch = X_batch.unsqueeze(0).expand(torch.Size((nb_imputation,))+X_batch.shape).flatten(0,1)
            if output_S.shape[-1] == 2 :
                mask = torch.ones_like(X_batch, dtype=torch.float32)

                mask_firstdim = mask.clone()
                mask_firstdim[:,0] = torch.zeros(X_batch.shape[0], dtype=torch.float32)
                

                X_batch_first_dim = self.impute_result(mask_firstdim, X_batch)
                Y_first_dim = classifier(X_batch_first_dim).reshape(nb_imputation, batch_size, self.nb_classes)


                mask_seconddim = mask.clone()
                mask_seconddim[:,1] = torch.zeros(X_batch.shape[0], dtype=torch.float32)
                X_batch_second_dim = self.impute_result(mask_seconddim, X_batch)
                Y_second_dim = classifier(X_batch_second_dim).reshape(nb_imputation, batch_size, self.nb_classes)

                                
                true_selection_firstdim = torch.mean(torch.std(Y_first_dim, dim=0), axis=-1)
                true_selection_seconddim = torch.mean(torch.std(Y_second_dim, dim=0), axis=-1)
        
                if k != nb_batch - 1 :
                    output_S[k*self.batch_size_train:(k+1)*self.batch_size_train,0] = true_selection_firstdim.detach().cpu().numpy()
                    output_S[k*self.batch_size_train:(k+1)*self.batch_size_train,1] = true_selection_seconddim.detach().cpu().numpy()                
                else :
                    output_S[k*self.batch_size_train:,0] = true_selection_firstdim.detach().cpu().numpy()
                    output_S[k*self.batch_size_train:,1] = true_selection_seconddim.detach().cpu().numpy()   
                

            else :
                raise NotImplementedError("Not implemented for more than 2 dimensions") 

        output_S = torch.tensor(output_S).type(torch.float32)
        if normalize :
            output_S /= torch.max(output_S)

        return output_S

class HypercubeDataset(ArtificialDataset):
    def __init__(self, nb_shape = None, nb_dim = None,  sigma=1.0, ratio_sigma = 0.25, prob_simplify=0.2,
                 nb_sample_train = 20, nb_sample_test = 20, give_index = False, batch_size_train = None,
                 noisy = None, use_cuda = False, centroids_path = None,
                 generate_new = False, save = False, generate_each_time = True,
                 exact_sel_dim = False, max_sel_dim = 2):

        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, batch_size_train = batch_size_train,
                 noisy = noisy, use_cuda = use_cuda,)

        self.nb_shape = nb_shape
        self.nb_dim = nb_dim
        self.sigma = sigma  
        self.nb_classes = 2
        print(f"Given sigma is {sigma}")
        self.prob_simplify = prob_simplify
        self.ratio_sigma = ratio_sigma
        self.index_neighboors_train = None
        self.index_neighboors_test = None


        if generate_new :
            print("Generate new data")
            if nb_shape is None or nb_dim is None :
                raise ValueError("Can't generate new dataset without information on dim and shapes")
            self.centroids, self.centroids_Y, self.S, self.shapes = gen_multivariate(n = nb_shape, d = nb_dim, max_sel_dim= max_sel_dim, sigma=sigma, prob_simplify=prob_simplify, exact_sel_dim=exact_sel_dim)
            if save:
                if centroids_path is None :
                    raise ValueError("Need a path to save the dataset")
                else :
                    if not os.path.exists(os.path.dirname(centroids_path)):
                        os.makedirs(os.path.dirname(centroids_path))
                    np.save(centroids_path, (self.centroids, self.centroids_Y, self.S, self.shapes))
        else :
            print("Load data from file")
            if not centroids_path.endswith(".npy"):
                centroids_path += ".npy"
            if (centroids_path is None) or (not os.path.exists(centroids_path)):
                raise FileNotFoundError(f"Did not find the file at {centroids_path}")
            self.centroids, self.centroids_Y, self.S, self.shapes = np.load(centroids_path, allow_pickle=True)

            nb_shape = len(self.shapes)
            nb_dim = self.centroids.shape[1]

            if self.nb_shape is not None and  self.nb_shape != nb_shape :
                raise ValueError(f"The number of shape wanted {self.nb_shape} is different from the number of shape loaded {nb_shape}")
                
            if self.nb_dim is not None and  self.nb_dim != nb_dim :
                raise ValueError(f"The number of dim wanted {self.nb_dim} is different from the number of dim loaded {nb_dim}")
        
        sigma = np.inf
        for k in range(len(self.centroids)):
            aux_centroids = np.reshape(self.centroids[k], (1,-1)).repeat(len(self.centroids)-1,axis=0)
            if k ==0 :
                sigma = min(self.sigma, np.min(np.max(np.abs(aux_centroids - self.centroids[1:]), axis=-1)))
            elif k==len(self.centroids)-1:
                sigma = min(self.sigma, np.min(np.max(np.abs(aux_centroids - self.centroids[:-1]), axis=-1)))
            else :
                aux_total = np.concatenate([self.centroids[:k], self.centroids[k+1:]], axis=0)
                sigma = min(self.sigma, np.min(np.max(np.abs(aux_centroids - aux_total), axis=-1)))
            
        self.sigma = sigma

        self.nb_shape = nb_shape
        self.nb_dim = nb_dim
        self.gaussian_noise = self.sigma * self.ratio_sigma
        self.centroids = torch.from_numpy(self.centroids).type(torch.float32)
        self.centroids_Y = torch.from_numpy(self.centroids_Y).type(torch.int64)
                    
                    
        print(f"Loaded {nb_shape} shapes with {nb_dim} dimensions")
        self.len_dim = np.array(list(map(lambda x: len(x), self.S)))

        print(f"Mean number of dim {np.mean(self.len_dim)}, Std number of dim {np.std(self.len_dim)}")
        print(f"sigma value is {self.sigma}")
        print(f"Noise in the dataset is {self.gaussian_noise}")



        if self.use_cuda :
            self.centroids = self.centroids.cuda()
            self.centroids_Y =self.centroids_Y.cuda()
        self.generate_each_time = generate_each_time

        self.S_exactdef = redraw_dependency(self.S, self.nb_dim)
        if self.use_cuda :
            self.S_exactdef = self.S_exactdef.cuda()
        self.X_train, self.Y_train, self.S_train_exactdef, self.X_test, self.Y_test, self.S_test_exactdef = generate_distribution(self.centroids, self.centroids_Y, self.S_exactdef, self.gaussian_noise, self.nb_sample_train, self.nb_sample_test)
        
        
        self.S_train_dataset_based_unnormalized = self.calculate_true_selection_variation(self.X_train,)
        self.S_test_dataset_based_unnormalized = self.calculate_true_selection_variation(self.X_test,)
        
        

        if self.generate_each_time :
            self.dataset_train = GeneratingTensorDataset(function=self.sample_function, len= self.batch_size_train, give_index = self.give_index)
        else :
            self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)

    def sample_function(self):
        index = np.random.randint(low=0, high = len(self.centroids))
        X = self.centroids[index] + torch.normal(torch.tensor(0.), std = self.gaussian_noise)
        Y = self.centroids_Y[index]
        return X, Y


    def find_hypercube_index(self, index):
        """ Find the hypercube which corresponds to the indexed point"""
        # TODO : FIND A BETTERWAY TO DO THIS ie add in gen multivariate a way to keep doing this
        for k, shape in enumerate(self.shapes):
            if index in shape :
                return k

    def find_imputation_centroid(self, current_point, hypercube, deleted_directions):
        """ Find the centroid for imputation depending on the hypercube index list and the deleted direction """
        list_index = []
        for index in hypercube :
            diff_index = np.where((current_point - self.X[index])!=0)[0]
            if diff_index in deleted_directions:
                list_index.append(index)
        return list_index

    def get_true_selection(self, index,  train_dataset = True):
        if not self.give_index :
            raise AttributeError("You need to give the index in the distribution if you want to use true Selection as input of a model")
        if train_dataset :
            true_S = self.new_S_train
        else :
            true_S = self.new_S_test
        true_S_value = true_S[index]
        return true_S_value

    def get_dependency(self, mask, value, index=None, dataset_type = None):
        batch_size, _ = value.shape
        nb_centroids, dim = self.centroids.shape
        
        # mask = mask.cpu()
        # value = value.cpu()
        
        mask_reshape = mask.unsqueeze(1).expand(batch_size, nb_centroids, dim)
        value_reshape = value.unsqueeze(1).expand(batch_size, nb_centroids, dim)
        centroids_reshape = self.centroids.unsqueeze(0).expand(batch_size, nb_centroids, dim)
        sigma = torch.ones_like(value_reshape) * self.gaussian_noise

        dependency = (((value_reshape - centroids_reshape)/sigma)**2)/2
        dependency = torch.where(mask_reshape == 0, torch.ones_like(dependency), dependency)
        dependency = - torch.sum(dependency, dim = -1)
        dependency = dependency - torch.logsumexp(dependency, dim = -1, keepdim = True)
        dependency = torch.exp(dependency)

        return dependency

    def get_true_output(self, value, mask = None, index=None, dataset_type = None):
        batch_size, _ = value.shape
        nb_centroids, dim = self.centroids.shape
        if mask is None :
            mask = torch.ones(value.shape)
        if value.is_cuda:
            mask = mask.cuda()    
    

        

        dependency = self.get_dependency(mask, value, index = None, dataset_type = None)
        aux_y = self.centroids_Y.unsqueeze(0).expand(batch_size, nb_centroids,)
        out_y = torch.sum(dependency * aux_y, dim = -1).unsqueeze(-1)
        out_y = torch.cat([out_y, torch.ones(batch_size, 1) - out_y], dim = -1)

    

        return out_y

    def impute_result(self, mask, value, index = None, dataset_type=None): 
        """ On part du principe que la value est complète mais c'est pas le cas encore, à gérer, sinon il faut transmettre l'index"""
        batch_size, _ = value.shape
        nb_centroids, dim = self.centroids.shape
        
        
        mask_reshape = mask.unsqueeze(1).expand(batch_size, nb_centroids, dim)
        value_reshape = value.unsqueeze(1).expand(batch_size, nb_centroids, dim)
        centroids_reshape = self.centroids.unsqueeze(0).expand(batch_size, nb_centroids, dim)
        sigma = torch.ones_like(value_reshape) * self.gaussian_noise


        dependency = (((value_reshape - centroids_reshape)/sigma)**2)/2
        dependency = torch.where(mask_reshape == 0, torch.ones_like(dependency), dependency)
        dependency = torch.exp(-torch.prod(dependency, axis=-1))  +1e-8
        dependency /= torch.sum(dependency, axis=-1, keepdim = True)

        index_resampling = torch.distributions.Multinomial(probs = dependency).sample().type(torch.int64)
        index_resampling = torch.argmax(index_resampling,axis=-1)

        wanted_centroids = self.centroids[index_resampling]
        sampled = wanted_centroids + torch.normal(torch.zeros_like(wanted_centroids), self.gaussian_noise).type(torch.float32)

        no_imputation_index = torch.where(torch.all(mask==1,axis=-1), True, False).unsqueeze(-1).expand(batch_size, dim)

        sampled = torch.where(no_imputation_index, value, sampled)

        return sampled

    # def calculate_local_neighboors(self, dataset_type = None, epsilon_selection = 0.5):
    #     if dataset_type == "train" :
    #         X = self.X_train
    #     elif dataset_type == "test" :
    #         X = self.X_test
    #     else :
    #         raise ValueError("dataset_type must be train or test")
    #     nb_centroids, dim = self.centroids.shape
    #     index_neighboors = []
    #     for x in X :
    #         x = x.unsqueeze(0)
    #         distance = torch.linalg.norm(X - x.expand(X.shape), dim = -1)
    #         index_neighboors.append(torch.where(distance < epsilon_selection)[0])
    #     if dataset_type=="train" :
    #         self.index_neighboors_train = index_neighboors
    #     else :
    #         self.index_neighboors_test = index_neighboors    
        




    # def local_impute(self, mask, value, index=None, dataset_type=None, epsilon_selection = 0.5, sigma_impute = 0.05):
    #     # TODO : Could be improved by adding some methods
    #     if index is None :
    #         raise ValueError("You need to give the index in the distribution if you want to use local imputation")

    #     batch_size, _ = value.shape
    #     if dataset_type == "train" :
    #         X = self.X_train
    #         if self.index_neighboors_train is None :
    #             self.calculate_local_neighboors(dataset_type = "train", epsilon_selection=epsilon_selection)
    #         index_neighboors = self.index_neighboors_train
    #     else :
    #         X = self.X_test
    #         if self.index_neighboors_test is None :
    #             self.calculate_local_neighboors(dataset_type = "test", epsilon_selection=epsilon_selection)
    #         index_neighboors = self.index_neighboors_test


    #     samples_list = []
    #     for i in range(len(value)):
    #         current_mask = mask[i]
    #         current_value = value[i]
    #         current_neighboors = X[index_neighboors[i]]
    #         current_value = current_value.unsqueeze(0).expand(current_neighboors.shape)
    #         current_mask = current_mask.unsqueeze(0).expand(current_neighboors.shape)
            
    #         dependency = (((current_value - current_neighboors)/sigma_impute)**2)/2
    #         dependency = torch.where(current_mask == 0, torch.ones_like(dependency), dependency)
    #         dependency = torch.exp(-torch.prod(dependency, axis=-1))  +1e-8
    #         dependency /= torch.sum(dependency, axis=-1, keepdim = True)

    #         index_resampling = torch.distributions.Multinomial(probs = dependency).sample().type(torch.int64)
    #         index_resampling = torch.argmax(index_resampling,axis=-1)

    #         wanted_centroid = current_neighboors[index_resampling]
    #         sampled = wanted_centroid + torch.normal(torch.zeros_like(wanted_centroid), self.gaussian_noise)
            
            






class LinearDataset(ArtificialDataset):
    def __init__(self, nb_sample_train = 1000, nb_sample_test = 1000, give_index = False, 
                 batch_size_train = None, use_cuda = False, noisy = None):
        
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, batch_size_train = batch_size_train,
                 noisy = noisy, use_cuda = use_cuda,)

        
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.batch_size_train = batch_size_train
        self.use_cuda = use_cuda
        self.nb_dim = 2
        self.nb_classes = 2

        nb_sample = nb_sample_test + nb_sample_train
        X = scipy.stats.uniform([-2.0,-2.0], [4,4]).rvs((nb_sample,2))
        Y = np.where(X[:,0]<-X[:,1], np.ones((nb_sample)), np.zeros((nb_sample))).astype(np.int64)

        self.X_train = torch.tensor(X[:nb_sample_train], dtype= torch.float32)
        self.Y_train = torch.tensor(Y[:nb_sample_train], dtype = torch.int64)
        
        
        self.X_test = torch.tensor(X[nb_sample_train:], dtype=torch.float32)
        self.Y_test = torch.tensor(Y[nb_sample_train:], dtype = torch.int64)
        

        self.S_train_exactdef = torch.ones_like(self.X_train)
        self.S_test_exactdef = torch.ones_like(self.X_test)


        self.S_train_dataset_based_unnormalized = self.calculate_true_selection_variation(self.X_train,)
        self.S_test_dataset_based_unnormalized = self.calculate_true_selection_variation(self.X_test,)
        
        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)
    

    def get_dim(self):
        return self.nb_dim
 
    def get_true_selection(self, index,  train_dataset = True):
        if not self.give_index :
            raise AttributeError("You need to give the index in the distribution if you want to use true Selection as input of a model")
        if train_dataset :
            true_S = self.new_S_train
        else :
            true_S = self.new_S_test

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

    def impute_result(self, mask, value, index = None, dataset_type=None): 
        uniform = torch.distributions.uniform.Uniform(-2.0,2.0)
        resampled_value = uniform.sample(value.shape).type(torch.float32)
        if value.is_cuda :
            resampled_value = resampled_value.cuda()
        
        try :
            sampled = torch.where(mask>0.5, value, resampled_value)
        except(RuntimeError):
            print(mask, value, resampled_value)
        return sampled



## Dataset based on standard gaussian

class StandardGaussianDataset(ArtificialDataset):
    def __init__(self, sigma=1.0, nb_sample_train = 20, nb_sample_test = 20, give_index = False, 
                 batch_size_train = None, use_cuda = False, ):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, batch_size_train = batch_size_train,
                 use_cuda = use_cuda,)
        self.sigma = sigma  
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.batch_size_train = batch_size_train
        self.use_cuda = use_cuda


    def get_true_selection(self, index,  train_dataset = True):
        if not self.give_index :
            raise AttributeError("You need to give the index in the distribution if you want to use true Selection as input of a model")
        if train_dataset :
            true_S = self.new_S_train
        else :
            true_S = self.new_S_test

        if index.is_cuda :
            true_S = true_S.cuda()
            
        true_S_value = true_S[index]
        return true_S_value

    def impute_result(self, mask, value, index = None, dataset_type=None): 
        normal_distrib = torch.distributions.normal.Normal(0,sigma)
        resampled_value = normal_distrib.sample()
        sampled = torch.where(mask == 1, value, resampled_value)
        return sampled


##=========================== XOR DATASET ========================================




def generate_XOR(sigma, nb_sample_train, nb_sample_test,):
    normal_distrib = torch.distributions.normal.Normal(0,sigma)
    X_train = normal_distrib.sample((nb_sample_train, 10,))
    X_test = normal_distrib.sample((nb_sample_test, 10,))
    total_X = torch.cat((X_train, X_test), axis=0)

    prob_test = torch.exp(total_X[:,0]*total_X[:,1])
    prob_test = 1/(1+prob_test)

    Y = torch.Bernoulli(prob_test).sample()
    Y_train,  Y_test = Y[:nb_sample_train], Y[nb_sample_train:]
    new_S_train = torch.cat([torch.ones((nb_sample_train, 2)),torch.zeros((nb_sample_train, 8))], axis=1)
    new_S_test = torch.cat([torch.ones((nb_sample_test, 2)),torch.zeros((nb_sample_test, 8))], axis=1)

    return X_train, Y_train, new_S_train, X_test, Y_test, new_S_test

class XOR(StandardGaussianDataset):
    def __init__(self, sigma=1.0, nb_sample_train = 20, nb_sample_test = 20, give_index = False, 
                 batch_size_train = None, use_cuda = False, ):

        super().__init__(sigma=sigma, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, 
                 batch_size_train = batch_size_train, use_cuda = use_cuda, )
        print(f"Given sigma is {self.sigma}")
        self.use_cuda = use_cuda
        self.X_train, self.Y_train, self.new_S_train, self.X_test, self.Y_test, self.new_S_test = generate_XOR(sigma = self.sigma, nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test,)
        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)

       

#### =============================== Orange Skin =====================================================



def generateOrangeSkin(sigma, nb_sample_train, nb_sample_test,):
    normal_distrib = torch.distributions.normal.Normal(0,sigma)
    X_train = normal_distrib.sample((nb_sample_train, 10,))
    X_test = normal_distrib.sample((nb_sample_test, 10,))
    total_X = torch.cat((X_train, X_test), axis=0)

    prob_test = torch.exp(torch.sum(total_X[:,:4]**2, axis = 1) - 4.0) 
    prob_test = 1/(1+prob_test)

    Y = torch.Bernoulli(prob_test).sample()
    Y_train,  Y_test = Y[:nb_sample_train], Y[nb_sample_train:]
    new_S_train = torch.cat([torch.ones((nb_sample_train, 4)),torch.zeros((nb_sample_train, 6))], axis=1)
    new_S_test = torch.cat([torch.ones((nb_sample_test, 4)),torch.zeros((nb_sample_test, 6))], axis=1)


    return X_train, Y_train, new_S_train, X_test, Y_test, new_S_test

class OrangeSkin(StandardGaussianDataset):
    def __init__(self, sigma=1.0, nb_sample_train = 20, nb_sample_test = 20, give_index = False, 
                 batch_size_train = None, use_cuda = False, ):

        super().__init__(sigma=sigma, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, 
                 batch_size_train = batch_size_train, use_cuda = use_cuda, )
        print(f"Given sigma is {self.sigma}")
        self.use_cuda = use_cuda
        self.X_train, self.Y_train, self.new_S_train, self.X_test, self.Y_test, self.new_S_test = generateOrangeSkin(sigma = self.sigma, nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test,)
        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)

       

#### =============================== Non-Linear Additive model =====================================================



def generateNonLinearAdditiveModel(sigma, nb_sample_train, nb_sample_test,):
    normal_distrib = torch.distributions.normal.Normal(0,sigma)
    X_train = normal_distrib.sample((nb_sample_train, 10,))
    X_test = normal_distrib.sample((nb_sample_test, 10,))
    total_X = torch.cat((X_train, X_test), axis=0)

    prob_test = torch.exp(-100 * torch.sin(0.2*total_X[:,0]) + abs(total_X[:,1]) + total_X[:,2] + torch.exp(-total_X[:,3])  - 2.4) # What is written is very different from the paper TODO : check
    prob_test = 1/(1+prob_test)

    Y = torch.Bernoulli(prob_test).sample()
    Y_train,  Y_test = Y[:nb_sample_train], Y[nb_sample_train:]
    new_S_train = torch.cat([torch.ones((nb_sample_train, 2)),torch.zeros((nb_sample_train, 8))], axis=1)
    new_S_test = torch.cat([torch.ones((nb_sample_test, 2)),torch.zeros((nb_sample_test, 8))], axis=1)


    return X_train, Y_train, new_S_train, X_test, Y_test, new_S_test

class NonLinearAdditiveModel(StandardGaussianDataset):
    def __init__(self, sigma=1.0, nb_sample_train = 20, nb_sample_test = 20, give_index = False, 
                 batch_size_train = None, use_cuda = False, ):

        super().__init__(sigma=sigma, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, 
                 batch_size_train = batch_size_train, use_cuda = use_cuda, )
        print(f"Given sigma is {self.sigma}")
        self.use_cuda = use_cuda
        self.X_train, self.Y_train, self.new_S_train, self.X_test, self.Y_test, self.new_S_test = generateNonLinearAdditiveModel(sigma = self.sigma, nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test,)
        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)



#### =============================== Non-Linear Multiplicative model =====================================================
  
def generateSwitchFeature(sigma, nb_sample_train, nb_sample_test,): # In the switch dataset, this is not a mixture of orange skin for the second gaussian, also they use 4th to 8th features TODO : check
    nb_sample_train_1 = nb_sample_train//2
    nb_sample_test_1 = nb_sample_test//2
    nb_sample_train_2 = nb_sample_train - nb_sample_train_1
    nb_sample_test_2 = nb_sample_test - nb_sample_test_1

    normal_distrib_1 = torch.distributions.normal.Normal(3,sigma)
    X_train_1 = normal_distrib_1.sample((nb_sample_train_1, 10,))
    X_test_1 = normal_distrib_1.sample((nb_sample_test_1, 10,))
    total_X_1 = torch.cat((X_train_1, X_test_1), axis=0)
    
    prob_test_1 = torch.exp(torch.sum(total_X_1[:,:4]**2, axis = 1) - 4.0) 
    prob_test_1 = 1/(1+prob_test_1)
    


    normal_distrib_2 = torch.distributions.normal.Normal(-3,sigma)
    X_train_2 = normal_distrib_2.sample((nb_sample_train_2, 10,))
    X_test_2 = normal_distrib_2.sample((nb_sample_test_2, 10,))
    total_X_2 = torch.cat((X_train_2, X_test_2), axis=0)
    
    prob_test_2 = torch.exp(torch.sum(total_X_2[:,4:-1]**2, axis = 1) - 4.0) 
    prob_test_2 = 1/(1+prob_test_2)

    X_train = torch.cat((X_train_1, X_train_2), axis=0)
    X_test = torch.cat((X_test_1, X_test_2), axis=0)
    
    prob_test = torch.cat((prob_test_1[:nb_sample_train_1], prob_test_2[:nb_sample_train_2],
                           prob_test_1[nb_sample_train_1:], prob_test_2[nb_sample_train_2:],), axis=0)
    
    Y = torch.Bernoulli(prob_test).sample()
    Y_train,  Y_test = Y[:nb_sample_train], Y[nb_sample_train:]
    
    new_S_train= torch.zeros((nb_sample_train, 10))
    new_S_train[:nb_sample_train_1, :4] = 1
    new_S_train[nb_sample_train_1:, 4:-1] = 0

    new_S_test= torch.zeros((nb_sample_test, 10))
    new_S_test[:nb_sample_test_1, :4] = 1
    new_S_test[nb_sample_test_1:, 4:-1] = 0

    return X_train, Y_train, new_S_train, X_test, Y_test, new_S_test
  

class SwitchFeature(StandardGaussianDataset):
    def __init__(self, sigma=1.0, nb_sample_train = 20, nb_sample_test = 20, give_index = False, 
                 batch_size_train = None, use_cuda = False, ):

        super().__init__(sigma=sigma, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, 
                 batch_size_train = batch_size_train, use_cuda = use_cuda, )
        print(f"Given sigma is {self.sigma}")
        self.use_cuda = use_cuda
        self.X_train, self.Y_train, self.new_S_train, self.X_test, self.Y_test, self.new_S_test = generateSwitchFeature(sigma = self.sigma, nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test,)
        self.dataset_train = TensorDatasetAugmented(self.X_train, self.Y_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.X_test, self.Y_test, give_index = self.give_index)
    
    def impute_result(self, mask, value, index = None, dataset_type=None):
        
        bern = torch.bernoulli(torch.full(mask.shape, fill_value=torch.tensor(0.5)))
        mean = bern * 3 + (1 - bern) * -3
        
        normal_distrib = torch.distributions.normal.Normal(mean,self.sigma)
        resampled_value = normal_distrib.sample()
        sampled = torch.where(mask == 1, value, resampled_value)
        return sampled


##### ENCAPSULATION :

class LoaderArtificial():
    def __init__(self, dataset, batch_size_train = 512, batch_size_test = 512, nb_sample_train = 10000, nb_sample_test=512*100, noisy = False, root_dir = None):

        self.dataset = dataset(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, noisy = noisy, batch_size_train = batch_size_train)
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
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
