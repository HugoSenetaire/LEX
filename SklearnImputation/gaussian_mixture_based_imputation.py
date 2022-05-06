import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture

def train_gmm(data, n_components, save_path):
    """ Training a Gaussian Mixture Model on the data using sklearn. """
    print("TRAINING FOR {} COMPONENTS".format(n_components))
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))

    try :
      data = data.numpy()
    except AttributeError :
      data = data
    gm = GaussianMixture(n_components=n_components, covariance_type='diag',)
    batch_size = data.shape[0]
    data_flatten = data.reshape(batch_size, -1)
    gm.fit(data_flatten)
    mu = gm.means_
    covariances = gm.covariances_
    weights = gm.weights_
    pkl.dump((weights, mu, covariances), open(save_path, "wb"))
    print("save at ", save_path)
    print(f"{n_components} components saved")


class GaussianMixtureBasedImputation(nn.Module):
  def __init__(self, imputation_network_weights_path, **kwargs) :
    super().__init__()
    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Weights path does not exist for the Gaussian Mixture at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
     weights, means, covariances = pkl.load(f)
    self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
    self.means = nn.Parameter(torch.tensor(means, dtype = torch.float32), requires_grad=False)
    self.covariances = nn.Parameter(torch.tensor(covariances, dtype = torch.float32), requires_grad=False)
    self.nb_centers = np.shape(means)[0]

  def get_dependency(self, data, mask, index = None):
      """ Using the data and the mask, to get the dependency of every point to the component of the mixture
       
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        dependency : torch.Tensor of shape (batch_size, self.nb_centers)
            Get dependency for all data and centers

      """
      batch_size = data.shape[0]
      other_dim = data.shape[1:] 


      wanted_shape = torch.Size((batch_size, self.nb_centers, *other_dim))
      wanted_shape_flatten = torch.Size((batch_size, self.nb_centers,np.prod(other_dim),))


      data_expanded = data.detach().unsqueeze(1).expand(wanted_shape).reshape(wanted_shape_flatten)
      mask_expanded = mask.detach().unsqueeze(1).expand(wanted_shape).reshape(wanted_shape_flatten)
      
      centers = self.means.unsqueeze(0).expand(wanted_shape_flatten)
      variance = self.covariances.unsqueeze(0).expand(wanted_shape_flatten)
      weights = self.weights.unsqueeze(0).expand(torch.Size((batch_size, self.nb_centers,)))


      dependency = -(data_expanded - centers)**2/2/variance - torch.log(variance)/2
      dependency = torch.sum(dependency* mask_expanded,axis=-1) + torch.log(weights)
      dependency[torch.where(torch.isnan(dependency))] = torch.zeros_like(dependency[torch.where(torch.isnan(dependency))]) #TODO : AWFUL WAY OF CLEANING THE ERROR, to change
      dependency_max, _ = torch.max(dependency, axis = -1, keepdim = True)
      dependency -= torch.log(torch.sum(torch.exp(dependency - dependency_max) + 1e-8, axis = -1, keepdim=True)) + dependency_max
      dependency = torch.exp(dependency)

      return dependency, wanted_shape
      
  def __call__(self, data, mask, index=None,):
    raise NotImplementedError


class GaussianMixtureImputation(GaussianMixtureBasedImputation):
  def __init__(self, imputation_network_weights_path, mean_imputation = False, **kwargs):
    super().__init__(imputation_network_weights_path, )
    self.mean_imputation = mean_imputation
    

  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        Note : This is just doing a single sample multiple imputation, maybe it might be quicker to allow for multiple imputation ?        
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        sampled : torch.Tensor of shape (batch_size, nb_category)
            Sampled tensor from the Gaussian Mixture

        """

    with torch.no_grad() :
      dependency, wanted_shape = self.get_dependency(data, mask, index)
      index_resampling = torch.distributions.Multinomial(probs = dependency).sample().type(torch.int64)
      index_resampling = torch.argmax(index_resampling,axis=-1)
      wanted_centroids = self.means[index_resampling]
      wanted_covariances = self.covariances[index_resampling]

      wanted_shape = data.shape
      if self.mean_imputation :
          sampled = wanted_centroids.reshape(wanted_shape)
      else :
          sampled = torch.normal(wanted_centroids, torch.sqrt(wanted_covariances)).type(torch.float32).reshape(wanted_shape)
    return sampled


class GaussianMixtureDatasetImputation(GaussianMixtureBasedImputation):
  def __init__(self, imputation_network_weights_path, data_to_impute, **kwargs):
    super().__init__(imputation_network_weights_path, )
    try :
      self.data_to_impute = torch.from_numpy(data_to_impute)
    except :
      self.data_to_impute = data_to_impute

    self.dependency_data_to_impute, _ = self.get_dependency(data_to_impute, mask = torch.ones_like(data_to_impute), index = None ) #Transformer en sparse ? 
    self.dependency_data_to_impute = self.dependency_data_to_impute / (torch.sum(self.dependency_data_to_impute, axis = 0, keepdim = True) + 1e-8)
    self.use_cuda = False
    
    

  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        Note : This is just doing a single sample multiple imputation, maybe it might be quicker to allow for multiple imputation ?        
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        sampled : torch.Tensor of shape (batch_size, nb_category)
            Sampled tensor from the dataset using the Gaussian Mixture dataset imputation

        """
    if not self.use_cuda and self.data.is_cuda :
        self.use_cuda = True
        self.data_to_impute = self.data_to_impute.cuda()
    batch_size = len(data)
    with torch.no_grad() :
      dependency, wanted_shape = self.get_dependency(data, mask, index)
      index_resampling_cluster = torch.distributions.Multinomial(probs = dependency).sample().type(torch.int64)
      index_resampling_cluster = torch.argmax(index_resampling_cluster,axis=-1)
      reverse_dependency = torch.t(self.dependency_data_to_impute[:, index_resampling_cluster]) #TOTEST/ Use scatter ?
      # reverse_dependency /= torch.sum(reverse_dependency, dim=1, keepdim=True) + 1e-8
      index_resampling_data = torch.argmax(torch.distributions.Multinomial(probs=reverse_dependency).sample().type(torch.int64), axis=-1)
      wanted_shape = data.shape
      sampled = self.data_to_impute[index_resampling_data.cpu()].reshape(wanted_shape)
      sampled = sampled.reshape(wanted_shape)
      if self.use_cuda :
        sampled = sampled.cuda()


    return sampled
