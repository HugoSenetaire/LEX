import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
import numpy as np
import torch
import copy

### SAMPLE_B REGULARIZATION :
class SampleB_regularization():
  def __init__(self):
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


class NetworkBasedRegularization():
  def __init__(self, network, use_cuda = False, to_train = False, deepcopy = False):
    self.network = network
    self.use_cuda = use_cuda
    self.to_train = to_train
    
    if deepcopy :
      self.network = copy.deepcopy(self.network)
    

    # if cuda :
      # self.network = self.network.cuda()
    self.network = self.network.cuda()
    if not to_train :
      for param in self.network.parameters():
          param.requires_grad = False

  def cuda(self):
    self.use_cuda = True
    self.network = self.network.cuda()

  def zero_grad(self):
    self.network.zero_grad()

  def eval(self):
    self.network.eval()

  def train(self):
    self.network.train()

  def parameters(self):
    return self.network.parameters()

  def __call__(self, data_imputed, data_expanded):
    raise NotImplementedError
    
### RECONSTRUCTION REGULARIZATION : 

class AutoEncoderReconstructionRegularization(NetworkBasedRegularization):

  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)
  

  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstruced = self.network(data_imputed)
    loss =  torch.nn.functional.mse_loss(data_reconstruced, data_expanded)
    return loss
### POST PROCESS REGULARIZATION :

class NetworkTransform(NetworkBasedRegularization):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = self.network(data_imputed)
    return data_reconstructed
  

class NetworkAdd(NetworkBasedRegularization):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)


  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = self.network(data_imputed)
    data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
    return data_imputed
  


class NetworkTransformMask(NetworkBasedRegularization):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = data_imputed * (1-sample_b) + self.network(data_imputed) * sample_b 
    return data_reconstructed

    

# class NetworkAddMask(NetworkBasedRegularization):
#   def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
#     super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

#   def __call__(self, data_expanded, data_imputed, sample_b):
#     data_reconstructed = self.network(data_imputed)
#     data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
#     return data_reconstructed