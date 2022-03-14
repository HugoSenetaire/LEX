import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture

def train_gmm(data, n_components, save_path):
    """ Training a Gaussian Mixture Model on the data using sklearn. """
    print("TRAINING FOR {} COMPONENTS".format(n_components))
    gm = GaussianMixture(n_components=n_components, covariance_type='diag',)
    gm.fit(data)
    mu = gm.means_
    covariances = gm.covariances_
    weights = gm.weights_
    pkl.dump((weights, mu, covariances), open(save_path, "wb"))
    print("save at ", save_path)
    print(f"{n_components} components saved")

class GaussianMixtureImputation(nn.Module):
  def __init__(self, imputation_network_weights_path, mean_imputation = False, **kwargs):
    super().__init__()
    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Weights path does not exist for the Gaussian Mixture at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
     weights, means, covariances = pkl.load(f)
    self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
    self.means = nn.Parameter(torch.tensor(means, dtype = torch.float32), requires_grad=False)
    self.covariances = nn.Parameter(torch.tensor(covariances, dtype = torch.float32), requires_grad=False)
    self.nb_centers = np.shape(means)[0]
    self.mean_imputation = mean_imputation
    

  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        
        Parameters:
        -----------
        data : torch.Tensor of shape (nb_imputation, batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        sampled : torch.Tensor of shape (nb_imputation, batch_size, nb_category)
            Sampled tensor from the Gaussian Mixture

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