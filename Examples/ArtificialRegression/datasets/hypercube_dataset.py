import torch
import numpy as np 
import os


from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented

torch.pi = torch.tensor(3.1415)


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

def generate_distribution_local(centroids_X, centroids_Y, optimal_S, sigma, nb_samples = 20):
    nb_point, nb_dim = centroids_X.shape

    augmented_X = centroids_X.unsqueeze(1).expand(-1, nb_samples, -1).flatten(0,1)
    optimal_S_reshaped = optimal_S.unsqueeze(1).expand(-1,nb_samples,-1).flatten(0,1)

    Y = centroids_Y.unsqueeze(-1).expand(-1, nb_samples).flatten(0,1)
    X = augmented_X + torch.normal(torch.zeros_like(augmented_X), std = sigma)

    # X = X.type(torch.float32)

    
    return X,Y, optimal_S_reshaped

def generate_distribution(centroids_X, centroids_Y, optimal_S, sigma, nb_sample_train = 20, nb_sample_test = 20):
  X_train, Y_train, optimal_S_train = generate_distribution_local(centroids_X, centroids_Y, optimal_S, sigma, nb_sample_train)
  X_test, Y_test, optimal_S_test = generate_distribution_local(centroids_X, centroids_Y, optimal_S, sigma, nb_sample_test) 

  return X_train, Y_train, optimal_S_train, X_test, Y_test, optimal_S_test


def redraw_dependency(S, nb_dim):
  nb_shape = len(S)
  optimal_S = torch.zeros((nb_shape, nb_dim))
  for k in range(len(S)):
    optimal_S[k, S[k]] = torch.ones(len(S[k]))

  return optimal_S
 



class HypercubeDataset(ArtificialDataset):
    def __init__(self, nb_shape = None, nb_dim = None, nb_classes=2,  sigma=1.0, ratio_sigma = 0.25, prob_simplify=0.2,
                 nb_sample_train = 20, nb_sample_test = 20, give_index = False,
                 noise_function = None,  centroids_path = None,
                 generate_new = False, save = False, generate_each_time = True,
                 exact_sel_dim = False, max_sel_dim = 2, **kwargs):

        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)

        self.nb_shape = nb_shape
        self.nb_dim = nb_dim
        self.sigma = sigma  
        assert(nb_classes == 2) ## TODO : Can change this to multiclass ?
        self.nb_classes = nb_classes
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




        self.generate_each_time = generate_each_time
        self.S_exactdef = redraw_dependency(self.S, self.nb_dim)

        self.data_train, self.target_train, self.S_train_exactdef, self.data_test, self.target_test, self.S_test_exactdef = generate_distribution(self.centroids, self.centroids_Y, self.S_exactdef, self.gaussian_noise, self.nb_sample_train, self.nb_sample_test)
        
        
        self.S_train_dataset_based_unnormalized = self.calculate_true_selection_variation(self.data_train,)
        self.S_test_dataset_based_unnormalized = self.calculate_true_selection_variation(self.data_test,)
        
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index, noise_function=self.noise_function)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index, noise_function=self.noise_function)

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
            true_S = self.optimal_S_train
        else :
            true_S = self.optimal_S_test
        true_S_value = true_S[index]
        return true_S_value

    def get_dependency(self, mask, value, index=None, dataset_type = None):
        batch_size, _ = value.shape
        nb_centroids, dim = self.centroids.shape
        
        
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
        if value.is_cuda:
            aux_y = aux_y.cuda()   
        out_y = torch.sum(dependency * aux_y, dim = -1).unsqueeze(-1)
        one_vector = torch.ones(batch_size, 1)
        if value.is_cuda :
            one_vector = one_vector.cuda()
        out_y = torch.cat([out_y, one_vector - out_y], dim = -1)

        return out_y

    def impute(self, value,  mask, index = None, dataset_type=None): 
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

        return sampled




