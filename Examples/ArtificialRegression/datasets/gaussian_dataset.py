import torch
import numpy as np

from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented
from .utils import f_prod, f_squaredsum, f_squaredsum2



def create_toeplitz_covariance(rho, dim=11):
    """
    Create a toeplitz covariance for a multivariate gaussian distribution

    Args:
        rho (float): correlation coefficient
        dim (int): dimension of the covariance

    Returns:
        torch.Tensor: covariance matrix
    """
    cov = torch.zeros(dim, dim)
    for k in range(0, dim):
        for i in range(0, dim):
            cov[i, k] = pow(rho, abs(i - k))
    return cov


def create_blockwise_covariance(block_dims, rho_blocks, dim=11):
    """
    Create a block wise covariance for a multivariate gaussian distribution.

    Args:
        block_dims (list): list of block index (Note that blocks can be far apart as the order fo dimension do not matter.)
        rho_blocks (list): list of correlation coefficients for each block
        dim (int): dimension of the covariance matrix

    Returns:
        torch.Tensor: blockwise covariance matrix
    """
    assert len(block_dims) == len(rho_blocks)
    cov = torch.zeros(dim, dim, dtype=torch.float32)
    for block_dim, rho_block in zip(block_dims, rho_blocks) :
        for i in block_dim :
            for j in block_dim :
                cov[i,j] = 1 if i==j else rho_block
    return cov



class GaussianDataset(ArtificialDataset):
    def __init__(self,
                mean,
                cov,
                covariance_type = 'spherical',
                nb_sample_train = 20,
                nb_sample_test = 20,
                dim_input=11,
                give_index = False,
                noise_function = None,
                train_seed = 0,
                test_seed = 1,
                **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function=noise_function, **kwargs)
        torch.manual_seed(train_seed)
        self.mean = mean
        self.cov = cov
        if isinstance(self.cov, int) or isinstance(self.cov, float):
            self.cov = torch.tensor(self.cov, dtype=torch.float32)
        self.covariance_type = covariance_type
        self.dim_input = dim_input


        if self.covariance_type == 'spherical' :
            assert len(self.cov.shape) == 0
        elif self.covariance_type == 'diagonal' :
            assert self.cov.shape[0] == self.dim_input
        elif self.covariance_type == 'full':
            assert self.cov.shape[0] == self.cov.shape[1]
            assert self.cov.shape[0] == self.dim_input
        else :
            raise ValueError("covariance_type not understood")

        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index

        self.X = self.generate_data()

    def generate_data(self):
        mean_vector = torch.full((self.dim_input,), fill_value= self.mean)
        if self.covariance_type == 'spherical' :
            normal_distrib = torch.distributions.normal.Normal(mean_vector,torch.full(size =(self.dim_input,), fill_value = self.cov))
        elif self.covariance_type == 'diagonal' :
            normal_distrib = torch.distributions.normal.Normal(mean_vector, self.cov)
        elif self.covariance_type == 'full':
            normal_distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean_vector, self.cov)
        else :
            raise NotImplementedError("covariance_type not implemented")
        sampled = normal_distrib.sample((self.nb_sample_train + self.nb_sample_test,))
        return sampled

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

    def impute(self, value,  mask, index = None, dataset_type=None): 
        if self.covariance_type == 'spherical' :
            normal_distrib = torch.distributions.normal.Normal(torch.full_like(input = value, fill_value= self.mean),torch.full_like(input =value, fill_value = self.cov))
        elif self.covariance_type == 'diagonal' :
            normal_distrib = torch.distributions.normal.Normal(torch.full_like(input = value, fill_value= self.mean), self.cov.unsqueeze(0).expand(value.shape))
        elif self.covariance_type == 'full':
            raise NotImplementedError("covariance_type not implemented")
        else :
            raise AttributeError("covariance_type not implemented")
        sampled = normal_distrib.sample()
        return sampled


  




#### =============================== Non-Linear Multiplicative model =====================================================
  
def generateSwitchFeature(sigma, nb_sample_train, nb_sample_test,): # In the switch dataset, this is not a mixture of orange skin for the second gaussian, also they use 4th to 8th features TODO : check
    nb_sample_train_1 = nb_sample_train//2
    nb_sample_test_1 = nb_sample_test//2
    nb_sample_train_2 = nb_sample_train - nb_sample_train_1
    nb_sample_test_2 = nb_sample_test - nb_sample_test_1

    normal_distrib_1 = torch.distributions.normal.Normal(3,sigma)
    data_train_1 = normal_distrib_1.sample((nb_sample_train_1, 10,))
    data_test_1 = normal_distrib_1.sample((nb_sample_test_1, 10,))
    total_X_1 = torch.cat((data_train_1, data_test_1), axis=0)
    
    prob_test_1 = torch.exp(torch.sum(total_X_1[:,:4]**2, axis = 1) - 4.0) 
    prob_test_1 = 1/(1+prob_test_1)
    


    normal_distrib_2 = torch.distributions.normal.Normal(-3,sigma)
    data_train_2 = normal_distrib_2.sample((nb_sample_train_2, 10,))
    data_test_2 = normal_distrib_2.sample((nb_sample_test_2, 10,))
    total_X_2 = torch.cat((data_train_2, data_test_2), axis=0)
    
    prob_test_2 = torch.exp(torch.sum(total_X_2[:,4:-1]**2, axis = 1) - 4.0) 
    prob_test_2 = 1/(1+prob_test_2)

    data_train = torch.cat((data_train_1, data_train_2), axis=0)
    data_test = torch.cat((data_test_1, data_test_2), axis=0)
    
    prob_test = torch.cat((prob_test_1[:nb_sample_train_1], prob_test_2[:nb_sample_train_2],
                           prob_test_1[nb_sample_train_1:], prob_test_2[nb_sample_train_2:],), axis=0)
    
    Y = torch.Bernoulli(prob_test).sample()
    target_train,  target_test = Y[:nb_sample_train], Y[nb_sample_train:]
    
    optimal_S_train= torch.zeros((nb_sample_train, 10))
    optimal_S_train[:nb_sample_train_1, :4] = 1
    optimal_S_train[nb_sample_train_1:, 4:-1] = 0

    optimal_S_test= torch.zeros((nb_sample_test, 10))
    optimal_S_test[:nb_sample_test_1, :4] = 1
    optimal_S_test[nb_sample_test_1:, 4:-1] = 0

    return data_train, target_train, optimal_S_train, data_test, target_test, optimal_S_test
  

class SwitchFeature(GaussianDataset):
    def __init__(self, sigma=1.0,
            nb_sample_train = 20,
            nb_sample_test = 20,
            give_index = False,
            noise_function = None,
            train_seed = 0,
            test_seed = 0,
            **kwargs):
        super().__init__(sigma=sigma,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        give_index = give_index,
                        noise_function= noise_function,
                        train_seed=train_seed,
                        test_seed= test_seed,
                        **kwargs)
        print(f"Given sigma is {self.sigma}")
        self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generateSwitchFeature(sigma = self.sigma,
                                                                                                                                                nb_sample_train = self.nb_sample_train,
                                                                                                                                                nb_sample_test = self.nb_sample_test,
                                                                                                                                            )
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)
    
    def impute(self, value,  mask, index = None, dataset_type=None):
        bern = torch.bernoulli(torch.full(mask.shape, fill_value=torch.tensor(0.5)))
        mean = bern * 3 + (1 - bern) * -3
        normal_distrib = torch.distributions.normal.Normal(mean,self.sigma)
        sampled = normal_distrib.sample()
        return sampled

#### ========================================= A FEW SIMPLE DATASET TO SEE THE EFFECT OF IMPUTATION ON THE PERFORMANCE ===========================================


class SimpleGaussianDataset(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                used_dim = 2,
                classification = True,
                epsilon_sigma = 0.3,
                dscale_regression = True,
                give_index = False,
                noise_function = None,
                train_seed = 0,
                test_seed = 1,
                **kwargs):

        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        give_index = give_index,
                        noise_function = noise_function,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        **kwargs)

        self.used_dim = used_dim
        
        self.classification = classification
        if self.classification :
            self.nb_classes = 2
        else :
            self.nb_classes = 1
        
        self.epsilon_sigma = epsilon_sigma
        assert self.used_dim <= self.dim_input

        fa, b_fa, sel = self.function(self.X, self.used_dim)
        
        if self.classification :
            Y = torch.rand(size = b_fa.shape)
            Y = torch.where(Y<b_fa, torch.ones_like(b_fa, dtype = torch.int64), torch.zeros_like(b_fa, dtype = torch.int64))
        else :
            if dscale_regression :
                Y = torch.distributions.Normal(b_fa,epsilon_sigma).sample()
            else :
                Y = torch.distributions.Normal(fa,epsilon_sigma).sample()


        self.data_train = self.X[:self.nb_sample_train,:]
        self.data_test = self.X[self.nb_sample_train:,:]
        self.target_train = Y[:self.nb_sample_train,]
        self.target_test = Y[nb_sample_train:,]
        self.optimal_S_train = sel[:nb_sample_train,:]
        self.optimal_S_test = sel[nb_sample_train:,:]
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)
    





class ExpProdGaussianDataset(SimpleGaussianDataset):
   def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                used_dim = 2,
                classification = True,
                epsilon_sigma = 0.3,
                dscale_regression = True,
                give_index = False,
                noise_function = None,
                train_seed = 0,
                test_seed = 0,
                **kwargs):
        self.function = f_prod
        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        used_dim = used_dim,
                        classification = classification,
                        epsilon_sigma = epsilon_sigma,
                        dscale_regression = dscale_regression,
                        give_index = give_index,
                        noise_function = noise_function,
                        train_seed = train_seed,
                        test_seed = test_seed,
                        **kwargs)

      



class ExpSquaredSumGaussianDataset(SimpleGaussianDataset):
     def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                used_dim = 2,
                classification = True,
                epsilon_sigma = 0.3,
                dscale_regression = True,
                give_index = False,
                noise_function = None,
                train_seed = 0,
                test_seed = 1,
                **kwargs):
        self.function = f_squaredsum
        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        used_dim = used_dim,
                        classification = classification,
                        epsilon_sigma = epsilon_sigma,
                        dscale_regression = dscale_regression,
                        give_index = give_index,
                        noise_function = noise_function,
                        train_seed = train_seed,
                        test_seed = test_seed,
                        **kwargs)

 

class ExpSquaredSumGaussianDatasetV2(SimpleGaussianDataset):
     def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                used_dim = 2,
                classification = True,
                epsilon_sigma = 0.3,
                dscale_regression = True,
                give_index = False,
                noise_function = None,
                train_seed = 0,
                test_seed = 1,
                **kwargs):
        self.function = f_squaredsum2
        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        used_dim = used_dim,
                        classification = classification,
                        epsilon_sigma = epsilon_sigma,
                        dscale_regression = dscale_regression,
                        give_index = give_index,
                        noise_function = noise_function,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        **kwargs)

    


class DiagGaussianDataset(GaussianDataset):
    def __init__(self, mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                give_index = False,
                noise_function = None,
                dim_input = 2,
                train_seed = 0,
                test_seed = 1,
                **kwargs):
        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        give_index = give_index,
                        noise_function = noise_function,
                        dim_input= dim_input,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        **kwargs) 

        self.nb_sample = self.nb_sample_test + self.nb_sample_train

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
    
