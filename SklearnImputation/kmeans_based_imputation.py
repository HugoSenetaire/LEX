import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans


def fit_auxiliary_kmeans(data, n_components,):
    """ 
    We just fit an auxiliary kmeans with random data to initialize it faster.
    Indeed, Kmeans need fitting to be initialized.
    """
    kmeans = KMeans(n_clusters=n_components,)
    batch_size = data.shape[0]
    data_flatten = data.reshape(batch_size, -1)
    random = np.random.rand(n_components, data_flatten.shape[1],)
    random = random.astype(np.float32)
    kmeans.fit(random)
    return kmeans

def train_kmeans(data, n_components, save_path):
    """ Training a KMEANS on the data using sklearn. """
    print("TRAINING FOR {} COMPONENTS".format(n_components))
    kmeans = KMeans(n_clusters=n_components,)
    batch_size = data.shape[0]
    data_flatten = data.reshape(batch_size, -1)
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))
    kmeans.fit(data_flatten)
    clusters_centers_ = kmeans.cluster_centers_
    pkl.dump((clusters_centers_), open(save_path, "wb"))
    print("save at ", save_path)
    print(f"{n_components} components saved")






class KmeansDatasetImputation(nn.Module):
  """
    Use a Kmeans on data to impute to get 
    Does SKLEARN Allows a choice of distance ?
  """
  def __init__(self, imputation_network_weights_path, data_to_impute, **kwargs):
    super().__init__()

    try :
      data_to_impute = data_to_impute.numpy()
    except AttributeError as e:
      print(e)

    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Centers path does not exist for the Kmeans at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
      centers = pkl.load(f).astype(np.float32)

    size_data_impute = data_to_impute.shape[0]
    self.nb_centers = centers.shape[0]
    self.kmeans = fit_auxiliary_kmeans(data_to_impute, self.nb_centers)  
    self.centers = nn.Parameter(torch.from_numpy(centers,), requires_grad=False)
    self.kmeans.cluster_centers_ = centers
    clusters_prediction = self.kmeans.predict(data_to_impute.reshape(size_data_impute, -1))
    self.dependency_data_to_impute = torch.from_numpy(clusters_prediction, )
    self.index_per_cluster = torch.unique(self.dependency_data_to_impute, return_inverse=True)[1]
    self.data_to_impute = torch.from_numpy(data_to_impute, )
    self.len_data_to_impute = len(self.data_to_impute)
    self.use_cuda = False


  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        Note : This is just doing a single sample multiple imputation.
        
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
    
    if not self.use_cuda and self.data.is_cuda :
      self.use_cuda = True
    with torch.no_grad() :
      batch_size = data.shape[0]
      other_dim = data.shape[1:] 


      wanted_shape = torch.Size((batch_size, self.nb_centers, *other_dim))
      wanted_shape_flatten = torch.Size((batch_size, self.nb_centers,np.prod(other_dim),))
      data_expanded = data.detach().unsqueeze(1).expand(wanted_shape).reshape(wanted_shape_flatten)
      mask_expanded = mask.detach().unsqueeze(1).expand(wanted_shape).reshape(wanted_shape_flatten)
      
      centers = self.centers.unsqueeze(0).expand(wanted_shape_flatten)

      distance = (data_expanded - centers)**2
      distance = torch.sum(distance * mask_expanded, dim=2)

      index_min = torch.argmin(distance, dim=1)
      probs = torch.zeros((batch_size, self.len_data_to_impute), dtype=torch.float32)
      impute_data_to_be_selected = self.index_per_cluster[index_min]
      probs[torch.arange(batch_size), impute_data_to_be_selected] = 1 #TOTEST
      probs = probs / (torch.sum(probs, dim=1, keepdim=True) + 1e-8)
      index_resampling = torch.distributions.Multinomial(probs = probs).sample().type(torch.int64) 
      index_resampling = torch.argmax(index_resampling, dim=1)
      sampled = self.data_to_impute[index_resampling.cpu()]
      wanted_shape = data.shape
      sampled = sampled.reshape(wanted_shape)
      if self.use_cuda :
        sampled = sampled.cuda()

  
    return sampled