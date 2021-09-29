import torch
import torchvision
from torch.utils.data import TensorDataset, Dataset, DataLoader
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




def bit_count(bin_subset):
    ''' Cardinality of the set corresponding to bin_subset. '''
    c = 0
    while bin_subset:
        bin_subset &= bin_subset - 1
        c += 1
    return c

def gen_multivariate(n, d, max_sel_dim = 2, sigma = 0.25, prob_simplify = 0.2, mix_shape=True):
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

    # print(X_real)
    return X_real, Y_real, S, shapes

def generate_distribution_local(centroids_X, centroids_Y, new_S, sigma, nb_samples = 20):
    nb_point, nb_dim = centroids_X.shape
    # plt.scatter(centroids_X[:,0], centroids_X[:,1])
    # plt.show()
    augmented_X = centroids_X.unsqueeze(1).expand(-1, nb_samples, -1).flatten(0,1)
    new_S_reshaped = new_S.unsqueeze(1).expand(-1,nb_samples,-1).flatten(0,1)
    # plt.scatter(augmented_X[:,0], augmented_X[:,1])
    # plt.show()


    Y = centroids_Y.unsqueeze(-1).expand(-1, nb_samples).flatten(0,1)
    X = augmented_X + torch.normal(torch.zeros_like(augmented_X), std = sigma)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()
    
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
 


class HypercubeDataset(Dataset):
    def __init__(self, nb_shape = None, nb_dim = None,  sigma=1.0, ratio_sigma = 0.25, prob_simplify=0.2,
                 nb_sample_train = 20, nb_sample_test = 20, give_index = False, batch_size_train = None,
                 noisy = None, use_cuda = False, centroids_path = None,
                 generate_new = False, save = False, generate_each_time = True):

        self.nb_shape = nb_shape
        self.nb_dim = nb_dim
        self.sigma = sigma  
        self.use_cuda = use_cuda
        self.prob_simplify = prob_simplify
        self.ratio_sigma = ratio_sigma
        self.gaussian_noise = self.sigma * self.ratio_sigma

        if generate_new :
            if nb_shape is None or nb_dim is None :
                raise ValueError("Can't generate new dataset without information on dim and shapes")
            self.centroids, self.centroids_Y, self.S, self.shapes = gen_multivariate(nb_shape, nb_dim, sigma=sigma, prob_simplify=prob_simplify)
            if save:
                if centroids_path is None :
                    raise ValueError("Need a path to save the dataset")
                else :
                    if not os.path.exists(os.path.dirname(centroids_path)):
                        os.makedirs(os.path.dirname(centroids_path))
                    np.save(centroids_path, (self.centroids, self.centroids_Y, self.S, self.shapes))
        else :
            if (centroids_path is None) or (not os.path.exists(centroids_path)):
                raise FileNotFoundError(f"Did not find the file at {centroids_path}")
            self.centroids, self.centroids_Y, self.S, self.shapes = np.load(centroids_path, allow_pickle=True)
            nb_shape = len(self.shapes)
            nb_dim = self.centroids.shape[1]

            if self.nb_shape is not None and  self.nb_shape != nb_shape :
                raise ValueError(f"The number of shape wanted {self.nb_shape} is different from the number of shape loaded {nb_shape}")
                
            if self.nb_dim is not None and  self.nb_dim != nb_dim :
                raise ValueError(f"The number of dim wanted {self.nb_dim} is different from the number of dim loaded {nb_dim}")
            
            print(f"Loaded {nb_shape} shapes with {nb_dim} dimensions")
        self.nb_shape = nb_shape
        self.nb_dim = nb_dim
        self.centroids = torch.from_numpy(self.centroids).type(torch.float32)
        self.centroids_Y = torch.from_numpy(self.centroids_Y).type(torch.int64)

        if self.use_cuda :
            self.centroids = self.centroids.cuda()
            self.centroids_Y =self.centroids_Y.cuda()


        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.batch_size_train = batch_size_train
        self.generate_each_time = generate_each_time

        self.new_S = redraw_dependency(self.S, self.nb_dim)
        if self.use_cuda :
            self.new_S = self.new_S.cuda()
        self.X_train, self.Y_train, self.new_S_train, self.X_test, self.Y_test, self.new_S_test = generate_distribution(self.centroids, self.centroids_Y, self.new_S, self.gaussian_noise, self.nb_sample_train, self.nb_sample_test)
        
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


 
    def compare_selection(self, mask, index = None, normalized = False, train_dataset = False):

        if train_dataset :
          current_S = self.new_S_train
        else :
          current_S = self.new_S_test

        if index is not None :
            true_masks = current_S[index]
        else :
            true_masks = current_S

      
        accuracy = torch.sum(1-torch.abs(true_masks - mask))
        proportion = torch.count_nonzero(torch.all(torch.abs(true_masks - mask) == 0,axis=-1))
        auc_score = sklearn.metrics.roc_auc_score(true_masks.flatten().detach().cpu().numpy(), mask.flatten().detach().cpu().numpy())
        auc_score = torch.tensor(auc_score).type(torch.float32)
        if normalized : 
            accuracy = accuracy/self.nb_dim/len(true_masks)
            proportion = proportion/len(true_masks)
        else :
            auc_score = auc_score*self.nb_dim*len(true_masks)

        return accuracy, proportion, auc_score

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
        sampled = wanted_centroids + torch.normal(torch.zeros_like(wanted_centroids), self.gaussian_noise)

        no_imputation_index = torch.where(torch.all(mask==1,axis=-1), True, False).unsqueeze(-1).expand(batch_size, dim)
        sampled = torch.where(no_imputation_index, value, sampled)

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
