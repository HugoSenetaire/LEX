from random import sample
import sys
import os

from numpy.core.fromnumeric import var
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
import numpy as np
import torch
import torch.nn as nn
import copy
from importlib import import_module
import matplotlib.pyplot as plt
import pandas as pd
from .vaeac import *
import inspect
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle as pkl
import numpy.ma as ma

#### UTILS 


def expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation, index = None):
    wanted_transform = torch.Size((nb_imputation,)) + data_expanded.shape
    data_imputed_expanded = data_imputed.unsqueeze(0).expand(wanted_transform).flatten(0,1)
    data_expanded_imputation = data_expanded.unsqueeze(0).expand(wanted_transform).flatten(0,1) 
    mask_expanded = sample_b.unsqueeze(0).expand(wanted_transform).flatten(0,1)
    if index is not None :
      wanted_transform_index = torch.Size((nb_imputation,)) + index.shape
      index_expanded = index.unsqueeze(0).expand(wanted_transform_index).flatten(0,1)
    else:
      index_expanded = None
    return data_imputed_expanded, data_expanded_imputation, mask_expanded, index_expanded






##### POST PROCESS :


### SAMPLE_B REGULARIZATION :
class SampleB_regularization(nn.Module):
  def __init__(self):
    super().__init__()
    self.to_train = False

  def __call__(self, data_expanded, sample_b):
    raise NotImplementedError


class SimpleB_Regularization(SampleB_regularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    if self.rate > np.random.random():
      sample_b = torch.ones(data_expanded.shape).cuda()
    
    return sample_b


  
class Less_Destruction_Regularization(SampleB_regularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    sample_b = torch.where(
      ((sample_b<0.5) * torch.rand(sample_b.shape, device = "cuda")>self.rate),
      torch.zeros(sample_b.shape,device = "cuda"),
      sample_b
    )
    return sample_b

class Complete_Inversion_Regularization(SampleB_regularization):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):

    sample_b = torch.where(
      (torch.rand(sample_b.shape, device = "cuda")>self.rate),
      1-sample_b,
      sample_b
    )

    return sample_b


##### MORE COMPLICATED IMPUTATION :
  

class NetworkBasedPostProcess(nn.Module):
  def __init__(self, network, deepcopy = False, to_train = False):
    super(NetworkBasedPostProcess, self).__init__()
    self.network = network
    self.multiple_imputation = False

    self.to_train = to_train
    if deepcopy :
      self.network = copy.deepcopy(self.network)
    self.network = self.network.cuda()

  def train(self):
    super(NetworkBasedPostProcess, self).train()
    if not self.to_train :
      self.network.eval()

  def __call__(self, data_expanded, data_imputed, sample_b,index = None):
    raise NotImplementedError



class MultipleImputation(nn.Module):
  def __init__(self, nb_imputation):
    super().__init__()
    self.multiple_imputation = True
    self.nb_imputation = nb_imputation



  def check_mode(self):
    if self.training :
      return self.nb_imputation
    else :
      return 1



class NetworkBasedMultipleImputation(NetworkBasedPostProcess, MultipleImputation):
  def __init__(self, network, to_train = False, deepcopy = False, nb_imputation = 3, use_cuda = True):
    NetworkBasedPostProcess.__init__(self, network, to_train=to_train, deepcopy=deepcopy)
    MultipleImputation.__init__(self, nb_imputation = nb_imputation)
    self.use_cuda = use_cuda

  def eval(self):
    NetworkBasedPostProcess.eval(self)
    MultipleImputation.eval(self)

  def train(self):
    NetworkBasedPostProcess.train(self)
    MultipleImputation.train(self)



### LOSS REGULARIZATION : 

class AutoEncoderReconstructionRegularization(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)
  
  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    data_reconstruced = self.network(data_imputed)
    loss =  torch.nn.functional.mse_loss(data_reconstruced, data_expanded)
    return loss
  
### POST PROCESS REGULARIZATION :

class NetworkTransform(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    data_reconstructed = self.network(data_imputed)
    return data_reconstructed, data_expanded, sample_b
  

class NetworkAdd(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)


  def __call__(self, data_expanded, data_imputed, sample_b, index = None,):
    data_reconstructed = self.network(data_imputed)
    data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
    return data_reconstructed, data_expanded, sample_b
  


class NetworkTransformMask(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    data_reconstructed = data_imputed * (1-sample_b) + self.network(data_imputed) * sample_b 
    return data_reconstructed, data_expanded, sample_b




class GaussianMixtureImputation(MultipleImputation):
  def __init__(self, imputation_network_weights_path, nb_imputation, **kwargs):
    super().__init__(nb_imputation)
    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Weights path does not exist for the Gaussian Mixture at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
     weights, means, covariances = pkl.load(f)
    self.weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False)
    self.means = torch.tensor(means, dtype = torch.float32, requires_grad=False)
    self.covariances = torch.tensor(covariances, dtype = torch.float32, requires_grad=False)
    self.nb_centers = np.shape(means)[0]

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    imputation_number = self.check_mode()

    batch_size = torch.Size((data_imputed.shape[0],))
    other_dim = data_imputed.shape[1:]

    data_expanded_flatten = data_expanded.flatten(1)
    sample_b_expanded_flatten = sample_b.flatten(1)
    data_imputed_flatten = data_imputed.flatten(1)

    
    wanted_shape_flatten = batch_size + torch.Size((self.nb_centers,np.prod(other_dim)))
    data_imputed_flatten = data_imputed_flatten.unsqueeze(1).expand(wanted_shape_flatten)
    sample_b_expanded_flatten = sample_b_expanded_flatten.unsqueeze(1).expand(wanted_shape_flatten)
  
    centers = self.means.unsqueeze(0).expand(wanted_shape_flatten)
    variance = self.covariances.unsqueeze(0).expand(wanted_shape_flatten)
    weights = self.weights.unsqueeze(0).expand(wanted_shape_flatten[:-1])
    if data_expanded.device.type == "cuda":
      centers = centers.cuda()
      variance = variance.cuda()
      weights = weights.cuda()


    # data_masked = ma.masked_array(data_imputed_flatten.detach().cpu().numpy(), mask = 1-sample_b_expanded_flatten)
    # centers_masked = ma.masked_array(centers.detach().cpu().numpy(), mask =1-sample_b_expanded_flatten)
    # variance_masked = ma.masked_array(variance.detach().cpu().numpy(), mask =1-sample_b_expanded_flatten)

    # dependency = -(data_masked - centers_masked)**2/2/variance_masked - ma.log(variance)/2
    # dependency_average = ma.expand_dims(ma.average(dependency, axis = -1),axis=-1)
    # dependency_sum = ma.exp(ma.sum(dependency - dependency_average, axis = -1) + dependency_average.squeeze())
    # dependency_sum /= ma.sum(dependency_sum, axis = -1, keepdims = True)
    # dependency = dependency_sum
    # print("=========================")
    dependency = -(data_imputed_flatten - centers)**2/2/variance - torch.log(variance)/2

    dependency = torch.sum(dependency* sample_b_expanded_flatten,axis=-1) + torch.log(weights)
    dependency[torch.where(torch.isnan(dependency))] = torch.zeros_like(dependency[torch.where(torch.isnan(dependency))]) #TODO : AWFUL WAY OF CLEANING THE ERROR, to change
    dependency_max, _ = torch.max(dependency, axis = -1, keepdim = True)
    dependency -= torch.log(torch.sum(torch.exp(dependency - dependency_max) + 1e-8, axis = -1, keepdim=True)) + dependency_max


    dependency = torch.exp(dependency)


    data_imputed, data_expanded, sample_b_expanded, index_expanded = expand_for_imputations(data_imputed, data_expanded, sample_b, imputation_number, index)
    wanted_shape = data_imputed.shape


    index_resampling = torch.distributions.Multinomial(probs = dependency).sample((imputation_number,)).type(torch.int64)
    index_resampling = torch.argmax(index_resampling,axis=-1)

    wanted_centroids = self.means[index_resampling]
    wanted_covariances = self.covariances[index_resampling]
    if data_expanded.device.type == "cuda":
      wanted_centroids = wanted_centroids.cuda()
      wanted_covariances = wanted_covariances.cuda()
    sampled = torch.normal(wanted_centroids, torch.sqrt(wanted_covariances)).type(torch.float32).reshape(wanted_shape)
    data_imputed_gm = sample_b_expanded * data_imputed + (1-sample_b_expanded) * sampled
    # fig, axs = plt.subplots(1,4, figsize = (20,5))
    # axs[0].imshow(data_imputed[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray",vmin = -0.1307, vmax = 1.0)
    # axs[1].imshow(sample_b_expanded[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray",vmin = 0.0, vmax = 1.0)
    # axs[2].imshow(data_imputed_gm[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray", vmin = -0.1307, vmax = 1.0)
    # axs[3].imshow(sampled[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray", vmin = -0.1307, vmax = 1.0)
    # plt.show()


    return data_imputed_gm, data_expanded, sample_b_expanded



class GaussianMixtureMeanImputation(MultipleImputation):
  def __init__(self, imputation_network_weights_path, nb_imputation, **kwargs):
    super().__init__(nb_imputation)
    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Weights path does not exist for the Gaussian Mixture at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
     weights, means, covariances = pkl.load(f)
    self.weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False)
    self.means = torch.tensor(means, dtype = torch.float32, requires_grad=False)
    self.covariances = torch.tensor(covariances, dtype = torch.float32, requires_grad=False)
    self.nb_centers = np.shape(means)[0]

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    imputation_number = self.check_mode()

    batch_size = torch.Size((data_imputed.shape[0],))
    other_dim = data_imputed.shape[1:]

    data_expanded_flatten = data_expanded.flatten(1)
    sample_b_expanded_flatten = sample_b.flatten(1)
    data_imputed_flatten = data_imputed.flatten(1)

    
    wanted_shape_flatten = batch_size + torch.Size((self.nb_centers,np.prod(other_dim)))
    data_imputed_flatten = data_imputed_flatten.unsqueeze(1).expand(wanted_shape_flatten)
    sample_b_expanded_flatten = sample_b_expanded_flatten.unsqueeze(1).expand(wanted_shape_flatten)
  
    centers = self.means.unsqueeze(0).expand(wanted_shape_flatten)
    variance = self.covariances.unsqueeze(0).expand(wanted_shape_flatten)
    weights = self.weights.unsqueeze(0).expand(wanted_shape_flatten[:-1])
    if data_expanded.device.type == "cuda":
      centers = centers.cuda()
      variance = variance.cuda()
      weights = weights.cuda()


    # data_masked = ma.masked_array(data_imputed_flatten.detach().cpu().numpy(), mask = 1-sample_b_expanded_flatten)
    # centers_masked = ma.masked_array(centers.detach().cpu().numpy(), mask =1-sample_b_expanded_flatten)
    # variance_masked = ma.masked_array(variance.detach().cpu().numpy(), mask =1-sample_b_expanded_flatten)

    # dependency = -(data_masked - centers_masked)**2/2/variance_masked - ma.log(variance)/2
    # dependency_average = ma.expand_dims(ma.average(dependency, axis = -1),axis=-1)
    # dependency_sum = ma.exp(ma.sum(dependency - dependency_average, axis = -1) + dependency_average.squeeze())
    # dependency_sum /= ma.sum(dependency_sum, axis = -1, keepdims = True)
    # dependency = dependency_sum
    # print("=========================")
    dependency = -(data_imputed_flatten - centers)**2/2/variance - torch.log(variance)/2

    dependency = torch.sum(dependency* sample_b_expanded_flatten,axis=-1) + torch.log(weights)
    dependency[torch.where(torch.isnan(dependency))] = torch.zeros_like(dependency[torch.where(torch.isnan(dependency))]) #TODO : AWFUL WAY OF CLEANING THE ERROR, to change
    dependency_max, _ = torch.max(dependency, axis = -1, keepdim = True)
    dependency -= torch.log(torch.sum(torch.exp(dependency - dependency_max) + 1e-8, axis = -1, keepdim=True)) + dependency_max


    dependency = torch.exp(dependency)


    data_imputed, data_expanded, sample_b_expanded, index_expanded = expand_for_imputations(data_imputed, data_expanded, sample_b, imputation_number, index)
    wanted_shape = data_imputed.shape


    index_resampling = torch.distributions.Multinomial(probs = dependency).sample((imputation_number,)).type(torch.int64)
    index_resampling = torch.argmax(index_resampling,axis=-1)

    wanted_centroids = self.means[index_resampling]
    # wanted_covariances = self.covariances[index_resampling]
    if data_expanded.device.type == "cuda":
      wanted_centroids = wanted_centroids.cuda()
      # wanted_covariances = wanted_covariances.cuda()
    sampled = wanted_centroids
    data_imputed_gm = sample_b_expanded * data_imputed + (1-sample_b_expanded) * sampled
    # fig, axs = plt.subplots(1,4, figsize = (20,5))
    # axs[0].imshow(data_imputed[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray",vmin = -0.1307, vmax = 1.0)
    # axs[1].imshow(sample_b_expanded[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray",vmin = 0.0, vmax = 1.0)
    # axs[2].imshow(data_imputed_gm[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray", vmin = -0.1307, vmax = 1.0)
    # axs[3].imshow(sampled[0].reshape((28,56)).cpu().detach().numpy(), cmap="gray", vmin = -0.1307, vmax = 1.0)
    # plt.show()


    return data_imputed_gm, data_expanded, sample_b_expanded

    
class DatasetBasedImputation(MultipleImputation):
  def __init__(self, dataset, nb_imputation):
    super().__init__(nb_imputation)
    self.dataset = dataset
    self.exist = hasattr(dataset, "impute_result") 
    if self.exist :
      self.nb_imputation = nb_imputation
    else :
      self.nb_imputation = 1
      print(f"There is no theoretical method for multiple imputation with {dataset}. DatasetBasedImputation is bypassed from now on.")
    
      

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    if self.exist :
      if self.training :
        dataset_type = "Train"
        imputation_number = self.nb_imputation
      else :
        imputation_number = 1
        dataset_type = "Test"

      data_imputed, data_expanded, sample_b_expanded, index_expanded = expand_for_imputations(data_imputed, data_expanded, sample_b, imputation_number, index)
      
      imputed_output = self.dataset.impute_result(mask = sample_b_expanded.clone().detach(), value = data_imputed.clone().detach(), index = index_expanded, dataset_type = dataset_type)
      imputed_output = sample_b_expanded * data_imputed + (1-sample_b_expanded) * imputed_output
      return imputed_output, data_expanded, sample_b_expanded
    else :
      return data_imputed, data_expanded, sample_b

  
def load_VAEAC(path_model):
  # import the module with the model networks definitions,
  # optimization settings, and a mask generator
  model_module = import_module(path_model + '.model')
  # build VAEAC on top of the imported networks
  model = VAEAC(
      model_module.reconstruction_log_prob,
      model_module.proposal_network,
      model_module.prior_network,
      model_module.generative_network
  )
  mask_generator = model_module.mask_generator
  sampler = model_module.sampler

  if  not os.path.exists(os.path.join(path_model, 'last_checkpoint.tar')):
    print("model has not been trained")
    raise NotImplementedError
  location = 'cuda'
  checkpoint = torch.load(os.path.join(path_model, 'last_checkpoint.tar'),
                          map_location=location)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  return model, sampler



class VAEAC_Imputation_DetachVersion(NetworkBasedMultipleImputation):
  def __init__(self, network, sampler, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(network= network, to_train=to_train, use_cuda=use_cuda, deepcopy=deepcopy)
    self.nb_imputation = nb_imputation
    self.sampler = sampler
    self.multiple_imputation = True
    


  def __call__(self, data_expanded, data_imputed, sample_b,index = None, show_output = False):
    batch = data_imputed
    masks = 1-sample_b
    init_shape = batch.shape[0]


    if torch.cuda.is_available():
        batch = batch.cuda()
        masks = masks.cuda()

    

  
    with torch.no_grad():
      samples_params = self.network.generate_samples_params(batch.detach(),
                                                    masks.detach(),
                                                    nb_imputation,
                                                    need_observed = False)

      img_samples = self.sampler(samples_params, multiple = True).flatten(0,1)

    
    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    new_data = img_samples.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 

    return new_data, data_expanded, sample_b

class VAEAC_Imputation_Renormalized(NetworkBasedMultipleImputation):
  def __init__(self, network, sampler, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(network= network, to_train=to_train, use_cuda=use_cuda, deepcopy=deepcopy)
    self.nb_imputation = nb_imputation
    self.sampler = sampler
    self.multiple_imputation = True
    


  def __call__(self, data_expanded, data_imputed, sample_b,index = None, show_output = False):
    batch = data_imputed
    masks = 1-sample_b
    init_shape = batch.shape[0]


    if torch.cuda.is_available():
        batch = batch.cuda()
        masks = masks.cuda()


  
    with torch.no_grad():
      batch = (batch * 0.3081 + 0.1307) / 255.
      # compute imputation distributions parameters
      samples_params = self.network.generate_samples_params(batch.detach(),
                                                    masks.detach(),
                                                    nb_imputation,
                                                    need_observed = False)
      # img_samples = []
      # for element in samples_params :
      img_samples = self.sampler(samples_params, multiple = True).flatten(0,1)

      img_samples = (img_samples * 255. -  0.1307)/0.3081

    
    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    new_data = img_samples.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 
    return new_data, data_expanded, sample_b





class MICE_imputation(MultipleImputation):
  def __init__(self, nb_imputation = 5):
    super().init(nb_imputation = nb_imputation)
    self.network = network


  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):

    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
      

    data_expanded_numpy = data_expanded.flatten(1).detach().cpu().numpy()
    mask = sample_b>0.5
    mask = mask.flatten(1).detach().cpu().numpy()
    data_expanded_numpy = np.where(mask, data_expanded_numpy, np.NaN)
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(data_expanded_numpy)
    data_imputed_output = []
    for k in range(nb_imputation):
      data_imputed_output.append(torch.tensor(imp.transform(data_expanded_numpy)).unsqueeze(1))

    data_imputed_output = torch.cat(data_imputed_output, axis=1).flatten(0,1)

    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    
    new_data = data_imputed_output.cuda() *  (1-sample_b) + data_expanded * sample_b 
    new_data = new_data

    return new_data, data_expanded, sample_b

  
class MICE_imputation_pretrained(MultipleImputation):
  def __init__(self,network, nb_imputation = 5):
    super().__init__(nb_imputation = nb_imputation)
    self.network = network

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    
    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
    


    data_expanded_numpy = data_expanded.flatten(1).detach().cpu().numpy()
    mask = sample_b>0.5
    mask = mask.flatten(1).detach().cpu().numpy()
    data_expanded_numpy = np.where(mask, data_expanded_numpy, np.NaN)
    
    data_imputed_output = []
    for k in range(nb_imputation):
      data_imputed_output.append(torch.tensor(self.network.transform(data_expanded_numpy)).unsqueeze(1))

    data_imputed_output = torch.cat(data_imputed_output, axis=1).flatten(0,1)

    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)

    new_data = data_imputed_output.cuda() *  (1-sample_b) + data_expanded * sample_b 
    new_data = new_data

    return new_data, data_expanded, sample_b





class MarkovChain():
  def __init__(self, train_loader, total_init_probability = None, total_transition_probability = None, use_cuda = True):
    aux = next(iter(train_loader))
    if len(aux) == 3 :
      example_x, target, index = aux
    else :
      example_x, target = aux
    # example_x, target = next(iter(train_loader))
    batch_size, output_dim, sequence_len = example_x.shape
    self.sequence_len = sequence_len
    self.output_dim = output_dim
    self.use_cuda=use_cuda
    if total_transition_probability is None or total_init_probability is None :
      self.train(train_loader)
    else :
       self.init_probability = total_init_probability
       self.transition_probability = total_transition_probability

    if self.use_cuda :
      self.init_probability = self.init_probability.cuda()
      self.transition_probability = self.transition_probability.cuda()

    self.log_init_probability = torch.log(self.init_probability)
    self.log_transition_probability = torch.log(self.transition_probability)

  def train(self, train_loader):
    self.init_probability = torch.zeros((self.output_dim))
    self.transition_probability = torch.zeros((self.output_dim, self.output_dim))
    if self.use_cuda :
      self.init_probability = self.init_probability.cuda()
      self.transition_probability = self.transition_probability.cuda()
    
    for aux in iter(train_loader):
      
      element = aux[0]
      if self.use_cuda :
        element=element.cuda()
      for sequence in element :
        sequence = sequence.transpose(0,1)
        self.init_probability += sequence[0]

        for k in range(1,len(sequence)):
          self.transition_probability[torch.argmax(sequence[k-1]).item()] += sequence[k]
    if self.use_cuda :
      self.init_probability = torch.tensor(self.init_probability/(torch.sum(self.init_probability) + 1e-8)).cuda()
    else : 
      self.init_probability = torch.tensor(self.init_probability/(torch.sum(self.init_probability) + 1e-8))

    for k in range(self.output_dim):
      self.transition_probability[k] /= (torch.sum(self.transition_probability[k]) + 1e-8)
    if self.use_cuda :
      self.transition_probability = torch.tensor(self.transition_probability, dtype = torch.float32).cuda()
    else :
      self.transition_probability = torch.tensor(self.transition_probability, dtype = torch.float32)

  def impute(self, data, masks, nb_imputation):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    if self.use_cuda :
      message = torch.zeros((batch_size, self.sequence_len, self.output_dim)).cuda()
    else :
      message = torch.zeros((batch_size, self.sequence_len, self.output_dim))


    masks_expanded = masks.unsqueeze(-1).expand(-1,self.sequence_len, self.output_dim)


    # Forward :

    message[:, 0, :] = torch.where(masks_expanded[:,0,:].type(torch.double) == 1, data[:, :, 0].type(torch.double), self.init_probability.type(torch.double).unsqueeze(0).expand(batch_size,-1)) # message I is arriving at i
    for i in range(1, self.sequence_len):
        message_previous = torch.matmul(message[:, i-1], self.transition_probability)
       
        message[:, i] = torch.where(masks_expanded[:,i,:] == 1, data[:, :,i], message_previous)
        message[:, i] = message[:, i]/(torch.sum(message[:, i], axis = -1, keepdim =True)+ 1e-8)


    # Backward : 
    output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len))
    masks_imputation = masks.unsqueeze(-2).expand((batch_size, nb_imputation, self.sequence_len))
    data_argmax_imputation = data_argmax.unsqueeze(-2).expand((batch_size, nb_imputation, self.sequence_len))

    message = message.unsqueeze(2).expand(-1, -1, nb_imputation, -1).clone() # batch size, sequence len, nb_imputation, output_dim
    dist = torch.distributions.categorical.Categorical(probs = message[:,-1])
    output_sample[:, :, -1] = torch.where(masks_imputation[:,:,-1] == 1, data_argmax_imputation[:,:,-1], dist.sample())

    for i in range(self.sequence_len-2, -1, -1):
      backward_message = torch.zeros((batch_size*nb_imputation, self.output_dim))
      aux_transition = self.transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, self.output_dim, self.output_dim)

      
      output_sample_masks =  torch.nn.functional.one_hot(output_sample[:, :, i+1].type(torch.int64), num_classes=self.output_dim).unsqueeze(-2).expand(-1,-1, self.output_dim, -1)>0.5
      if self.use_cuda :
        output_sample_masks = output_sample_masks.cuda()

      aux_transition = torch.masked_select(aux_transition, output_sample_masks).reshape(batch_size, nb_imputation, self.output_dim)
      message[:,i,:,:] *=aux_transition
      message[:,i,:,:] /= (torch.sum(message[:,i,:,:], axis=-1).unsqueeze(-1).expand(-1,-1, self.output_dim)+1e-8)

      aux_message = (torch.sum(message[:,i,:,:], axis=-1)==1).unsqueeze(-1).expand(-1,-1,self.output_dim)
      replace_vector = torch.ones(aux_message.shape)/self.output_dim
      if self.use_cuda:
        replace_vector = replace_vector.cuda()
      
      
      message[:,i,:,:] = torch.where(aux_message, message[:,i,:,:], replace_vector)
      dist = torch.distributions.categorical.Categorical(probs=message[:,i,:,:])
      output_sample[:, :, i] = torch.where(masks_imputation[:,:,i] == 1, data_argmax_imputation[:,:,i],dist.sample())
    
    output_sample = torch.nn.functional.one_hot(output_sample.type(torch.int64),num_classes=self.output_dim).transpose(-1, -2)
    
    
    return output_sample




# Markov chain imputation

class MarkovChainImputation(MultipleImputation):
  def __init__(self, markov_chain, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(nb_imputation=nb_imputation)
    self.markov_chain = markov_chain    
    self.use_cuda = use_cuda


  def __call__(self, data_expanded, data_imputed, sample_b, index = None, show_output = False):

    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1

    with torch.no_grad():

      output = self.markov_chain.impute(data_expanded, sample_b, nb_imputation = nb_imputation)

      if self.use_cuda == True :
        output = output.cuda()

  
    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)


    output = output.reshape(data_expanded.shape)
    new_data = output.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 
    return new_data, data_expanded, sample_b





## HMM Imputation :


import tqdm
class HMM():
  def __init__(self, train_loader, hidden_dim, nb_iter = 10, nb_start= 5, use_cuda = True, 
               train_hmm = True, save_weights=True, path_weights = None,
               ):
    aux = next(iter(train_loader))
    if len(aux) == 3:
      data, target, index = aux
    else :
      data, target = aux
      index = None
      
    batch_size, output_dim, sequence_len = data.shape
    self.sequence_len = sequence_len
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.nb_iter = nb_iter
    self.nb_start = nb_start
    self.use_cuda = use_cuda
    self.name = f"HMM_{self.output_dim}_{self.hidden_dim}_{self.sequence_len}"
    self.save_weights = save_weights
    self.train_hmm = train_hmm
    self.path_weights = path_weights

    
    if self.path_weights is not None :
      self.complete_path_weights = os.path.join(self.path_weights, self.name)
      self.path_init_weights = os.path.join(self.complete_path_weights, "init_weights.pt")
      self.path_transition_weights = os.path.join(self.complete_path_weights, "transition_weights.pt")
      self.path_emission_weights = os.path.join(self.complete_path_weights, "emission_weights.pt")

    else :
      if self.save_weights :
        print(f"No path to save weights for {self.name}")
        self.save_weights = False



    if not (os.path.exists(self.path_init_weights) and os.path.exists(self.path_transition_weights) and os.path.exists(self.path_emission_weights)):
      if not self.train_hmm:
        print("Need_training because no weights are stored there")
      self.init_probability = None
      self.transition_probability = None
      self.emission_probability = None
      self.train_hmm = True
    else :
      self.init_probability = torch.load(self.path_init_weights)
      self.transition_probability = torch.load(self.path_transition_weights)
      self.emission_probability = torch.load(self.path_emission_weights)
      self.train_hmm = train_hmm
      if self.use_cuda :
        self.init_probability = self.init_probability.cuda()
        self.transition_probability = self.transition_probability.cuda()
        self.emission_probability = self.emission_probability.cuda()

    if self.train_hmm:
      self.train(train_loader, nb_iter=self.nb_iter, nb_start=self.nb_start)
    else :
      self.save_weights = False
    
    if self.save_weights :
      print(f"Weights will be saved at {self.complete_path_weights}")
      if not os.path.exists(self.complete_path_weights):
        os.makedirs(self.complete_path_weights)
      torch.save(self.init_probability, self.path_init_weights)
      torch.save(self.transition_probability, self.path_transition_weights)
      torch.save(self.emission_probability, self.path_emission_weights)



  def forward(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))
    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim))
    if self.use_cuda :
      message = message.cuda()
    emission_probability_transpose = self.emission_probability.transpose(0,1)

    auxiliary_ones =  torch.ones(message[:,0,:].shape, dtype = torch.float32)
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()

    # Forward :
    message[:, 0, :] = self.init_probability.unsqueeze(0).expand(batch_size,-1)
    emission_message = torch.matmul(data[:,:,0], emission_probability_transpose)
    message[:, 0, :] *= torch.where(masks_expanded[:,0,:] == 1, emission_message, auxiliary_ones) # message I is arriving at i
    message[:, 0, :] /= torch.sum(message[:,0,:],axis=-1, keepdim = True)
    for i in range(1, self.sequence_len):
        message[:, i] = torch.matmul(message[:, i-1], self.transition_probability)
        emission_message = torch.matmul(data[:,:,i].type(torch.float32), emission_probability_transpose)
        emission_message_masked = torch.where(masks_expanded[:,i,:] == 1, emission_message, auxiliary_ones)
        message[:, i] *= emission_message_masked
        message[:, i] /= torch.sum(message[:,i,:],axis=-1, keepdim = True)

    return message



  def backward(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))
    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim), dtype=torch.float32)
    emission_probability_transpose = self.emission_probability.transpose(0,1)
    transition_probability_expand = self.transition_probability.unsqueeze(0).expand(batch_size, self.hidden_dim, self.hidden_dim)
    auxiliary_ones = torch.ones(message[:,0,:].shape)
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()
      message = message.cuda()

    # Backward :
    message[:, -1, :] = torch.ones(message[:,-1].shape)/self.hidden_dim
  
    for i in range(self.sequence_len-2, -1, -1):
      emission_part = torch.matmul(data[:,:,i+1], emission_probability_transpose)
      emission_part = torch.where(masks_expanded[:,i+1]==1, emission_part, auxiliary_ones)
      previous_message = (emission_part*message[:,i+1,:]).unsqueeze(-2).expand(-1,self.hidden_dim, -1)
      message[:, i, :] = torch.sum(transition_probability_expand* previous_message, axis=-1)

    return message



  def backward_sample_hidden(self, data, masks, message, nb_imputation):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)

    output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len))
    masks_imputation = masks.unsqueeze(-2).expand((-1, nb_imputation, -1))
    data_argmax_imputation = data_argmax.unsqueeze(-2).expand((-1, nb_imputation, -1))

    message = message.unsqueeze(2).expand(-1, -1, nb_imputation, -1).clone() # batch size, sequence len, nb_imputation, hidden_dim ?
    dist = torch.distributions.categorical.Categorical(probs = message[:,-1])
    output_sample[:, :, -1] = dist.sample()
    if self.use_cuda:
      output_sample = output_sample.cuda()

    for i in range(self.sequence_len-2, -1, -1):
      aux_transition = self.transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, self.hidden_dim, self.hidden_dim)
      output_sample_masks =  torch.nn.functional.one_hot(output_sample[:, :, i+1].type(torch.int64), num_classes=self.hidden_dim).unsqueeze(-2).expand(-1,-1, self.hidden_dim, -1)>0.5
      aux_transition = torch.masked_select(aux_transition, output_sample_masks).reshape(batch_size, nb_imputation, self.hidden_dim)
      message[:,i,:,:] *=aux_transition
      message[:,i,:,:] /= torch.sum(message[:,i,:,:], axis=-1).unsqueeze(-1).expand(-1,-1, self.hidden_dim)
      dist = torch.distributions.categorical.Categorical(probs=message[:,i,:,:])
      output_sample[:, :, i] = dist.sample()
    return output_sample

  def backward_maximum(self, data, masks, message):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    nb_imputation = 1

    output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len))
    masks_imputation = masks[:,0,:].unsqueeze(-2).expand((-1, nb_imputation, -1))
    data_argmax_imputation = data_argmax.unsqueeze(-2).expand((-1, nb_imputation, -1))
    
    message = message.unsqueeze(2).expand(-1, -1, nb_imputation, -1).clone() # batch size, sequence len, nb_imputation, output_dim
    dist = torch.distributions.categorical.Categorical(probs = message[:,-1])
    output_sample[:, :, -1] = torch.argmax(message[:,-1], axis=-1)

    for i in range(self.sequence_len-2, -1, -1):
      aux_transition = self.transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, self.hidden_dim, self.hidden_dim)
      output_sample_masks =  torch.nn.functional.one_hot(output_sample[:, :, i+1].type(torch.int64), num_classes=self.hidden_dim).unsqueeze(-2).expand(-1,-1, self.hidden_dim, -1)>0.5
      aux_transition = torch.masked_select(aux_transition, output_sample_masks).reshape(batch_size, nb_imputation, self.hidden_dim)
      message[:,i,:,:] *=aux_transition
      message[:,i,:,:] /= torch.sum(message[:,i,:,:], axis=-1).unsqueeze(-1).expand(-1,-1, self.hidden_dim)
      output_sample[:, :, i] = torch.argmax(message[:,i,:,:], axis=-1)
    return output_sample



  def sample_observation(self, data, mask, latent):
    """ Given latent, data and mask, this function will sample the observations according to the latent state and the emission probability if mask == 1 or will put the data instead
        
        mask : torch.tensor Shape (batch_size, sequence_len)
        data : torch.tensor Shape (batch_size, output_dim, sequence_len)
        latent: torch.tensor Shape (batch_size, nb_imputation, sequence_len)     
    """
    batch_size = data.shape[0]
    nb_imputation = latent.shape[1]
    output_sample_latent = torch.nn.functional.one_hot(latent.type(torch.int64), num_classes=self.hidden_dim) # Shape (batch_size, nb_imputation, sequence_len, hidden_dim)
    data_expanded = data.unsqueeze(1).expand(-1, latent.shape[1], -1, -1).transpose(-1,-2) # Shape (batch_size, nb_imputation, sequence_len, output_dim)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(data_expanded.shape) # Shape (batch_size, nb_imputation, sequence_len, output_dim)

    probs = torch.matmul(output_sample_latent.type(torch.float32), self.emission_probability)
    probs = torch.where(mask_expanded == 1, data_expanded, probs)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    observations = dist.sample()

    return observations


  def train(self, train_loader, nb_iter=10, nb_start = 5):
    dic_best = {}
    dic_best["init"] = self.init_probability 
    dic_best["transition"] = self.transition_probability
    dic_best["emission"] = self.emission_probability
    if self.init_probability is None or self.transition_probability is None or self.emission_probability is None :
      dic_best["likelihood"] = -float("inf")
    else :
      log_likelihood = self.likelihood_total(train_loader)
      dic_best["likelihood"] = log_likelihood

    for num_start in range(nb_start):
      self.init_probability = torch.rand((self.hidden_dim))
      self.init_probability /=torch.sum(self.init_probability)
      self.transition_probability = torch.rand((self.hidden_dim, self.hidden_dim))
      self.transition_probability /=torch.sum(self.transition_probability, axis=-1, keepdim=True)
      self.emission_probability = torch.rand((self.hidden_dim, self.output_dim))
      self.emission_probability /=torch.sum(self.emission_probability, axis=-1, keepdim=True)

      if self.use_cuda :
        self.init_probability =self.init_probability.cuda()
        self.transition_probability = self.transition_probability.cuda()
        self.emission_probability = self.emission_probability.cuda()

      for num_iter in tqdm.tqdm(range(nb_iter)):
        total_element = 0
        gamma = torch.zeros((self.sequence_len, self.hidden_dim), dtype=torch.float32)
        gamma_aux = torch.zeros((self.sequence_len, self.hidden_dim, self.output_dim), dtype=torch.float32)
        zeta = torch.zeros((self.sequence_len-1, self.hidden_dim,self.hidden_dim), dtype=torch.float32)
        gamma_limited = torch.zeros((self.sequence_len-1, self.hidden_dim), dtype = torch.float32)

        if self.use_cuda :
          gamma = gamma.cuda()
          gamma_aux = gamma_aux.cuda()
          zeta = zeta.cuda()
          gamma_limited = gamma_limited.cuda()

        for batch_number, aux in enumerate(iter(train_loader)):
            element = aux[0]
            batch_size, _, _ = element.shape
            mask = torch.ones(torch.argmax(element,axis=1).shape)
            if self.use_cuda :
              mask = mask.cuda()
              element = element.cuda()
            element_transpose = element.transpose(1,2) # batch_size, sequence_len, num_hidden

            message_forward = self.forward(element, mask)
            message_forward = message_forward.unsqueeze(-1).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            message_forward_limited = message_forward[:,:-1]

            emission_message = torch.matmul(element_transpose, self.emission_probability.transpose(-1,-2)).unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            emission_message_limited = emission_message[:,1:]

            transition_expanded = self.transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, self.sequence_len-1, self.hidden_dim, self.hidden_dim)
            message_backward = self.backward(element, mask).unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            message_backward_limited = message_backward[:,1:]


            zeta_aux =  message_forward_limited* emission_message_limited * transition_expanded * message_backward_limited
            zeta += torch.sum(zeta_aux, axis=0)
            gamma_limited += torch.sum(torch.sum(zeta_aux, axis=-1),axis=0)

            gamma_current = torch.sum(message_forward*emission_message*message_backward, axis=-1)
            gamma += torch.sum(gamma_current, axis=0)


            element_transpose_expanded = element_transpose.unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.output_dim)
            gamma_aux += torch.sum(element_transpose_expanded*gamma_current.unsqueeze(-1).expand(-1,-1,-1,self.output_dim), axis=0)
            total_element+=batch_size
        




        # gamma_limited = gamma[:-1]
        self.transition_probability = torch.sum(zeta, axis=0)/torch.sum(gamma_limited, axis=0).unsqueeze(-1).expand(self.hidden_dim, self.hidden_dim)
        self.init_probability = gamma[0]/torch.sum(gamma[0],axis=0,keepdim=True)
        self.emission_probability = torch.sum(gamma_aux, axis=0)/torch.sum(gamma,axis=0).unsqueeze(-1).expand(self.hidden_dim, self.output_dim)


     

      log_likelihood = self.likelihood_total(train_loader)
      print(f"\n Likelihood after iteration {num_start} is {log_likelihood}")



      if log_likelihood > dic_best["likelihood"]:
        dic_best["likelihood"] = log_likelihood
        dic_best["init"] = self.init_probability.clone()
        dic_best["transition"] = self.transition_probability.clone()
        dic_best["emission"] = self.emission_probability.clone()







    self.init_probability = dic_best["init"].clone()
    self.transition_probability = dic_best["transition"].clone()
    self.emission_probability = dic_best["emission"].clone()



    
  def calculate_likelihood(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))
    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim))
    emission_probability_transpose = self.emission_probability.transpose(0,1)
    auxiliary_ones = torch.ones(message[:,0,:].shape, dtype = torch.float32)/self.hidden_dim
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()
      message = message.cuda()


    # Forward :
    current_message = self.init_probability.unsqueeze(0).expand(batch_size,-1).clone()
    # print(data, emission_probability_transpose)
    # print(masks)
    emission_message = torch.matmul(data[:,:,0], emission_probability_transpose)
    current_message *= torch.where(masks_expanded[:,0,:] == 1, emission_message, auxiliary_ones) # message I is arriving at i
    # message[:, 0, :] /= torch.sum(message[:,0,:],axis=-1, keepdim = True)
    for i in range(1, self.sequence_len):
        previous_message = current_message
        current_message = torch.matmul(previous_message, self.transition_probability)
        emission_message = torch.matmul(data[:,:,i].type(torch.float32), emission_probability_transpose)
        emission_message_masked = torch.where(masks_expanded[:,i,:] == 1, emission_message, auxiliary_ones)
        current_message *= emission_message_masked
        # message[:, i] /= torch.sum(message[:,i,:],axis=-1, keepdim = True)

    proba = current_message.sum(axis=-1)
    # print("Proba", proba)

    return proba    
  
  def likelihood_total(self, train_loader):
      log_likelihood = torch.tensor(0.)
      nb_element = torch.tensor(0.)
      if self.use_cuda :
        log_likelihood = log_likelihood.cuda()
      for batch_number, aux in enumerate(iter(train_loader)):
        batch_size, output_dim, sequence_len = aux[0].shape
        element = aux[0]
        masks = torch.ones(torch.argmax(element,-2).shape)
        if self.use_cuda :
          element = element.cuda()
          masks = masks.cuda()
        log_likelihood += torch.sum(torch.log(self.calculate_likelihood(element, masks) + 1e-8))
        nb_element += batch_size
      
      log_likelihood -= torch.log(nb_element)
      return log_likelihood


  def impute(self, data, masks, nb_imputation):
    message_forward = self.forward(data, masks)
    latent = self.backward_sample_hidden(data, masks, message_forward, nb_imputation)
    output_total = self.sample_observation(data, masks, latent)
    output_total = torch.nn.functional.one_hot(output_total.squeeze(),self.output_dim).transpose(-1,-2)
    return output_total


  

def log_matmul(M1, M2):
  torch.exp(M1, M2)

def op_plus_eq(X1, X2):
  return torch.logsumexp(torch.cat([X1.unsqueeze(0), X2.unsqueeze(0)],axis=0),axis=0)

class HMMLog():
  def __init__(self, train_loader, hidden_dim, nb_iter = 10, nb_start= 5, use_cuda = True, 
               train_hmm = True, save_weights=True, path_weights = None,
               ):
    aux = next(iter(train_loader))
    if len(aux) == 3:
      data, target, index = aux
    else :
      data, target = aux
      index = None
      
    batch_size, output_dim, sequence_len = data.shape
    self.sequence_len = sequence_len
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.nb_iter = nb_iter
    self.nb_start = nb_start
    self.use_cuda = use_cuda
    self.name = f"HMM_{self.output_dim}_{self.hidden_dim}_{self.sequence_len}"
    self.train_hmm = train_hmm
    self.save_weights = save_weights


    self.path_weights = path_weights
    if self.path_weights is not None :
      self.complete_path_weights = os.path.join(self.path_weights, self.name)
      self.path_init_weights = os.path.join(self.complete_path_weights, "init_weights.pt")
      self.path_transition_weights = os.path.join(self.complete_path_weights, "transition_weights.pt")
      self.path_emission_weights = os.path.join(self.complete_path_weights, "emission_weights.pt")

    else :
      if self.save_weights :
        print(f"No path to save weights for {self.name}")
        self.save_weights = False



    if not (os.path.exists(self.path_init_weights) and os.path.exists(self.path_transition_weights) and os.path.exists(self.path_emission_weights)):
      if not self.train_hmm:
        print("Need_training because no weights are stored there")
      self.init_probability = None
      self.transition_probability = None
      self.emission_probability = None
      self.train_hmm = True
    else :
      self.init_probability = torch.load(self.path_init_weights)
      self.transition_probability = torch.load(self.path_transition_weights)
      self.emission_probability = torch.load(self.path_emission_weights)

      if self.use_cuda :
        self.init_probability = self.init_probability.cuda()
        self.transition_probability = self.transition_probability.cuda()
        self.emission_probability = self.emission_probability.cuda()

      self.log_init_probability = torch.log(self.init_probability)
      self.log_transition_probability = torch.log(self.transition_probability)
      self.log_emission_probability = torch.log(self.emission_probability)

      self.train_hmm = train_hmm



    if self.train_hmm:
       self.train(train_loader, nb_iter=self.nb_iter, nb_start=self.nb_start)
    
    
    if self.save_weights :
      print(f"Weights will be saved at {self.complete_path_weights}")
      if not os.path.exists(self.complete_path_weights):
        os.makedirs(self.complete_path_weights)
      torch.save(self.init_probability, self.path_init_weights)
      torch.save(self.transition_probability, self.path_transition_weights)
      torch.save(self.emission_probability, self.path_emission_weights)



   


  def forward(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))
    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim))
    if self.use_cuda :
      message = message.cuda()
    log_emission_probability_transpose = self.log_emission_probability.transpose(0,1)

    auxiliary_ones = torch.log(torch.ones(message[:,0,:].shape, dtype = torch.float32))
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()

    # Forward :
    message[:, 0, :] = self.log_init_probability.unsqueeze(0).expand(batch_size,-1)
    emission_message = torch.matmul(data[:,:,0], log_emission_probability_transpose)
    message[:, 0, :] += torch.where(masks_expanded[:,0,:] == 1, emission_message, auxiliary_ones) # message I is arriving at i
    message[:, 0, :] -= torch.logsumexp(message[:,0,:],axis=-1, keepdim = True)
    for i in range(1, self.sequence_len):
        message[:, i] = torch.matmul(message[:, i-1], self.log_transition_probability)
        emission_message = torch.matmul(data[:,:,i].type(torch.float32), log_emission_probability_transpose)
        emission_message_masked = torch.where(masks_expanded[:,i,:] == 1, emission_message, auxiliary_ones)
        message[:, i] += emission_message_masked
        message[:, i] -= torch.logsumexp(message[:,i,:],axis=-1, keepdim = True)

    return message



  def backward(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))

    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim), dtype=torch.float32)
    log_emission_probability_transpose = self.log_emission_probability.transpose(0,1)
    log_transition_probability_expand = self.log_transition_probability.unsqueeze(0).expand(batch_size, self.hidden_dim, self.hidden_dim)
    auxiliary_ones = torch.log(torch.ones(message[:,0,:].shape))
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()
      message = message.cuda()

    # Backward :
    message[:, -1, :] = torch.log(torch.ones(message[:,-1].shape)/self.hidden_dim)
  
    for i in range(self.sequence_len-2, -1, -1):
      emission_part = torch.matmul(data[:,:,i+1], log_emission_probability_transpose)
      emission_part = torch.where(masks_expanded[:,i+1]==1, emission_part, auxiliary_ones)
      previous_message = (emission_part+message[:,i+1,:]).unsqueeze(-2).expand(-1,self.hidden_dim, -1)
      message[:, i, :] = torch.logsumexp(log_transition_probability_expand + previous_message, axis=-1)

    return message



  def backward_sample_hidden(self, data, masks, message, nb_imputation):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)

    output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len))
    masks_imputation = masks.unsqueeze(-2).expand((-1, nb_imputation, -1))
    data_argmax_imputation = data_argmax.unsqueeze(-2).expand((-1, nb_imputation, -1))

    message = message.unsqueeze(2).expand(-1, -1, nb_imputation, -1).clone() # batch size, sequence len, nb_imputation, hidden_dim ?
    dist = torch.distributions.categorical.Categorical(probs = torch.exp(message[:,-1]))
    output_sample[:, :, -1] = dist.sample()
    if self.use_cuda:
      output_sample = output_sample.cuda()

    for i in range(self.sequence_len-2, -1, -1):
      aux_transition = self.log_transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, self.hidden_dim, self.hidden_dim)
      output_sample_masks =  torch.nn.functional.one_hot(output_sample[:, :, i+1].type(torch.int64), num_classes=self.hidden_dim).unsqueeze(-2).expand(-1,-1, self.hidden_dim, -1)>0.5
      aux_transition = torch.masked_select(aux_transition, output_sample_masks).reshape(batch_size, nb_imputation, self.hidden_dim)
      message[:,i,:,:] +=aux_transition
      message[:,i,:,:] -= torch.logsumexp(message[:,i,:,:], axis=-1).unsqueeze(-1).expand(-1,-1, self.hidden_dim)
      dist = torch.distributions.categorical.Categorical(probs=torch.exp(message[:,i,:,:]))
      output_sample[:, :, i] = dist.sample()
    return output_sample

  def backward_maximum(self, data, masks, message):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    nb_imputation = 1

    output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len))
    masks_imputation = masks.unsqueeze(-2).expand((-1, nb_imputation, -1))
    data_argmax_imputation = data_argmax.unsqueeze(-2).expand((-1, nb_imputation, -1))
    
    message = message.unsqueeze(2).expand(-1, -1, nb_imputation, -1).clone() # batch size, sequence len, nb_imputation, output_dim
    dist = torch.distributions.categorical.Categorical(probs = torch.exp(message[:,-1]))
    output_sample[:, :, -1] = torch.argmax(message[:,-1], axis=-1)

    for i in range(self.sequence_len-2, -1, -1):
      aux_transition = self.log_transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, self.hidden_dim, self.hidden_dim)
      output_sample_masks =  torch.nn.functional.one_hot(output_sample[:, :, i+1].type(torch.int64), num_classes=self.hidden_dim).unsqueeze(-2).expand(-1,-1, self.hidden_dim, -1)>0.5
      aux_transition = torch.masked_select(aux_transition, output_sample_masks).reshape(batch_size, nb_imputation, self.hidden_dim)
      message[:,i,:,:] +=aux_transition
      message[:,i,:,:] -= torch.logsumexp(message[:,i,:,:], axis=-1).unsqueeze(-1).expand(-1,-1, self.hidden_dim)
      output_sample[:, :, i] = torch.argmax(message[:,i,:,:], axis=-1)
    return output_sample



  def sample_observation(self, data, mask, latent):
    """ Given latent, data and mask, this function will sample the observations according to the latent state and the emission probability if mask == 1 or will put the data instead
        
        mask : torch.tensor Shape (batch_size, sequence_len)
        data : torch.tensor Shape (batch_size, output_dim, sequence_len)
        latent: torch.tensor Shape (batch_size, nb_imputation, sequence_len)     
    """
    batch_size = data.shape[0]
    nb_imputation = latent.shape[1]
    output_sample_latent = torch.nn.functional.one_hot(latent.type(torch.int64), num_classes=self.hidden_dim) # Shape (batch_size, nb_imputation, sequence_len, hidden_dim)
    data_expanded = data.unsqueeze(1).expand(-1, latent.shape[1], -1, -1).transpose(-1,-2) # Shape (batch_size, nb_imputation, sequence_len, output_dim)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(data_expanded.shape) # Shape (batch_size, nb_imputation, sequence_len, output_dim)

    probs = torch.matmul(output_sample_latent.type(torch.float32), self.emission_probability)
    probs = torch.where(mask_expanded == 1, data_expanded, probs)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    observations = dist.sample()

    return observations


  def train(self, train_loader, nb_iter=25, nb_start = 1):
    dic_best ={}
    dic_best["init"] = self.init_probability 
    dic_best["transition"] = self.transition_probability
    dic_best["emission"] = self.emission_probability
    if self.init_probability is None or self.transition_probability is None or self.emission_probability is None :
      dic_best["likelihood"] = -float("inf")
    else :
      log_likelihood = self.likelihood_total(train_loader)
      dic_best["likelihood"] = log_likelihood


    for num_start in range(nb_start):
      self.log_init_probability = torch.log(torch.rand((self.hidden_dim)))
      self.log_init_probability -=torch.logsumexp(self.log_init_probability, dim=-1, keepdim=True)
      self.log_transition_probability = torch.log(torch.rand((self.hidden_dim, self.hidden_dim)))
      self.log_transition_probability -=torch.logsumexp(self.log_transition_probability, -1, keepdim=True)
      self.log_emission_probability = torch.log(torch.rand((self.hidden_dim, self.output_dim)))
      self.log_emission_probability -=torch.logsumexp(self.log_emission_probability, -1, keepdim=True)

      if self.use_cuda :
        self.log_init_probability = self.log_init_probability.cuda()
        self.log_transition_probability = self.log_transition_probability.cuda()
        self.log_emission_probability = self.log_emission_probability.cuda()

      for num_iter in tqdm.tqdm(range(nb_iter)):
        total_element = 0
        log_gamma = torch.zeros((self.sequence_len, self.hidden_dim), dtype=torch.float32)
        log_gamma_aux = torch.zeros((self.sequence_len, self.hidden_dim, self.output_dim), dtype=torch.float32)
        log_zeta = torch.zeros((self.sequence_len-1, self.hidden_dim,self.hidden_dim), dtype=torch.float32)
        log_gamma_limited = torch.zeros((self.sequence_len-1, self.hidden_dim), dtype = torch.float32)

        gamma = torch.zeros((self.sequence_len, self.hidden_dim), dtype=torch.float32)
        gamma_aux = torch.zeros((self.sequence_len, self.hidden_dim, self.output_dim), dtype=torch.float32)
        zeta = torch.zeros((self.sequence_len-1, self.hidden_dim,self.hidden_dim), dtype=torch.float32)
        gamma_limited = torch.zeros((self.sequence_len-1, self.hidden_dim), dtype = torch.float32)

        if self.use_cuda :
          gamma = gamma.cuda()
          gamma_aux = gamma_aux.cuda()
          zeta = zeta.cuda()
          gamma_limited = gamma_limited.cuda()

        for batch_number, aux in enumerate(iter(train_loader)):
            element = aux[0]
            batch_size, _, _ = element.shape
            mask = torch.ones(torch.argmax(element,axis=1).shape)
            if self.use_cuda :
              mask = mask.cuda()
              element = element.cuda()
            element_transpose = element.transpose(1,2) # batch_size, sequence_len, num_hidden

            message_forward = self.forward(element, mask)
            message_forward = message_forward.unsqueeze(-1).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            message_forward_limited = message_forward[:,:-1]

            emission_message = torch.matmul(element_transpose, self.log_emission_probability.transpose(-1,-2)).unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            emission_message_limited = emission_message[:,1:]

            transition_expanded = self.log_transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, self.sequence_len-1, self.hidden_dim, self.hidden_dim)
            message_backward = self.backward(element, mask).unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.hidden_dim)
            message_backward_limited = message_backward[:,1:]


            total_element+=batch_size

            zeta_aux =  torch.exp(message_forward_limited+emission_message_limited+transition_expanded+message_backward_limited)
            zeta += torch.sum(zeta_aux, axis=0)
            gamma_limited += torch.sum(torch.sum(zeta_aux, axis=-1),axis=0)
            gamma_current = torch.sum(torch.exp(message_forward+emission_message+message_backward), axis=-1)
            gamma += torch.sum(gamma_current, axis=0)
            element_transpose_expanded = element_transpose.unsqueeze(-2).expand(batch_size, self.sequence_len, self.hidden_dim, self.output_dim)
            gamma_aux += torch.sum(element_transpose_expanded*gamma_current.unsqueeze(-1).expand(-1,-1,-1,self.output_dim), axis=0)
        


        self.transition_probability = torch.sum(zeta, axis=0)/torch.sum(gamma_limited, axis=0).unsqueeze(-1).expand(self.hidden_dim, self.hidden_dim)
        self.init_probability = gamma[0]/torch.sum(gamma[0],axis=0,keepdim=True)
        self.emission_probability = torch.sum(gamma_aux, axis=0)/torch.sum(gamma,axis=0).unsqueeze(-1).expand(self.hidden_dim, self.output_dim)

        
        self.log_transition_probability = torch.log(self.transition_probability)
        self.log_init_probability = torch.log(self.init_probability)
        self.log_emission_probability = torch.log(self.emission_probability)


      log_likelihood = self.likelihood_total(train_loader)
      print(f"\n Likelihood after iteration {num_start} is {log_likelihood}")



      if log_likelihood > dic_best["likelihood"]:
        dic_best["likelihood"] = log_likelihood
        dic_best["init"] = self.init_probability.clone()
        dic_best["transition"] = self.transition_probability.clone()
        dic_best["emission"] = self.emission_probability.clone()
    




    self.init_probability = dic_best["init"].clone()
    self.transition_probability = dic_best["transition"].clone()
    self.emission_probability = dic_best["emission"].clone()
    self.log_transition_probability = torch.log(self.transition_probability)
    self.log_init_probability = torch.log(self.init_probability)
    self.log_emission_probability = torch.log(self.emission_probability)




    
  def calculate_likelihood(self, data, masks):
    batch_size = data.shape[0]
    data_argmax = torch.argmax(data, axis=-2)
    masks_expanded = masks.unsqueeze(-1).expand((-1, -1, self.hidden_dim))

    message = torch.zeros((batch_size, self.sequence_len, self.hidden_dim))
    emission_probability_transpose = self.log_emission_probability.transpose(0,1)
    auxiliary_ones = torch.log(torch.ones(message[:,0,:].shape, dtype = torch.float32)/self.hidden_dim)
    if self.use_cuda :
      auxiliary_ones = auxiliary_ones.cuda()
      message = message.cuda()


    # Forward :
    current_message = self.log_init_probability.unsqueeze(0).expand(batch_size,-1).clone()
    emission_message = torch.matmul(data[:,:,0], emission_probability_transpose)
    current_message += torch.where(masks_expanded[:,0,:] == 1, emission_message, auxiliary_ones) # message I is arriving at i
    for i in range(1, self.sequence_len):
        previous_message = current_message
        current_message = torch.matmul(previous_message, self.transition_probability)
        emission_message = torch.matmul(data[:,:,i].type(torch.float32), emission_probability_transpose)
        emission_message_masked = torch.where(masks_expanded[:,i,:] == 1, emission_message, auxiliary_ones)
        current_message += emission_message_masked

    proba = torch.exp(torch.logsumexp(current_message,axis=-1))
    return proba   


    
  def likelihood_total(self, train_loader):
      log_likelihood = torch.tensor(0.)
      nb_element = torch.tensor(0.)
      if self.use_cuda :
        log_likelihood = log_likelihood.cuda()
      for batch_number, aux in enumerate(iter(train_loader)):
        batch_size, output_dim, sequence_len = aux[0].shape
        element = aux[0]
        masks = torch.ones(torch.argmax(element,-2).shape)
        if self.use_cuda :
          element = element.cuda()
          masks = masks.cuda()
        log_likelihood += torch.sum(torch.log(self.calculate_likelihood(element, masks) + 1e-8))
        nb_element += batch_size
      
      log_likelihood -= torch.log(nb_element)
      return log_likelihood

  def impute(self, data, masks, nb_imputation):
    message_forward = self.forward(data, masks)
    latent = self.backward_sample_hidden(data, masks, message_forward, nb_imputation)
    output_total = self.sample_observation(data, masks, latent)
    output_total = torch.nn.functional.one_hot(output_total.squeeze(),self.output_dim).transpose(-1,-2)
    return output_total


class HMMimputation(MultipleImputation):
  def __init__(self, hmm, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(nb_imputation=nb_imputation)
    self.hmm = hmm    
    self.use_cuda = use_cuda


  def __call__(self, data_expanded, data_imputed, sample_b, index = None, show_output = False):

    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1

    with torch.no_grad():

      output = self.hmm.impute(data_expanded, sample_b, nb_imputation = nb_imputation)

      if self.use_cuda == True :
        output = output.cuda()

  
    _, data_expanded, sample_b, _ = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)


    output = output.reshape(data_expanded.shape)
    new_data = output.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 
    return new_data, data_expanded, sample_b