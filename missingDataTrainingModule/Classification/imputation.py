import sys
import os

from scipy.sparse import data
from .post_process_imputation import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
import torch
import torch.nn as nn

def prepare_process(input_process):
  """ Utility function to make sure that the input_process is a list of function or None"""
  if input_process is None:
    return None
  else :
    if input_process is not list :
      input_process = [input_process]
    input_process = nn.ModuleList(input_process)

    return input_process

class Imputation(nn.Module):
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,
             ):
    super().__init__()
    self.isRounded = isRounded
    self.input_size = input_size
    self.add_mask = add_mask
    self.reconstruction_reg = prepare_process(reconstruction_reg)
    self.post_process_regularization = prepare_process(post_process_regularization)
    self.sample_b_reg = prepare_process(sample_b_reg)

    
    self.nb_imputation = 1
    if self.post_process_regularization is not None :
      for element in self.post_process_regularization :
        if element.multiple_imputation :
          self.nb_imputation *= element.nb_imputation
    self.use_cuda = False
  
  def has_constant(self):
    return False
 
  def has_rate(self):
    return False


  def round_sample(self, sample_b):
    """ Round sample so that we only have a discrete mask. Unadvised as the training becomes very unstable"""
    if self.isRounded :
      sample_b_rounded = torch.round(sample_b)
      return sample_b_rounded #N_expectation, batch_size, channels, size:...
    else :
      return sample_b

  def cuda(self):
    super().cuda()
    self.use_cuda = True
            
  def __str__(self):
    """ Make sure of what is printed when everything is in it """
    string = str(type(self)).split(".")[-1].replace("'","").strip(">")
    if self.has_constant():
      string +="_"+ str(self.cste)
    if self.has_rate():
      string +=f"_rate_{self.rate}"
    return string

  def imputation_reconstruction(self, data_expanded, data_imputed, sample_b, index = None):
    loss_reconstruction = torch.zeros((1))
    if self.use_cuda :
      loss_reconstruction = loss_reconstruction.cuda()
    if self.reconstruction_reg is not None :
      data_imputed = self.imputation_function(data_expanded, sample_b, index = index)
      for process in self.reconstruction_reg :
          loss_reconstruction += process(data_expanded, data_imputed, sample_b)
    return loss_reconstruction

  
  def add_mask_method(self, data_imputed, sample_b):
    if len(sample_b.shape)>2:
      sample_b_aux = sample_b[:,0].unsqueeze(1)
    else :
      sample_b_aux = sample_b
    return torch.cat([data_imputed, sample_b_aux], axis =1)


  def post_process(self, data_expanded, data_imputed, sample_b, index = None):
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        data_imputed, data_expanded, sample_b = process(data_expanded, data_imputed, sample_b, index = index)
    if self.add_mask:
      data_imputed = self.add_mask_method(data_imputed, sample_b)
    return data_imputed

  def sample_b_regularization(self, data_expanded, sample_b):
    if self.sample_b_reg is None :
      return sample_b
    else :
      for process in self.sample_b_reg :
        sample_b = process(data_expanded, sample_b)
      

      return sample_b


  def imputation_function(self, data_expanded, sample_b):
    raise NotImplementedError

  def forward(self, data_expanded, sample_b, index = None):
    sample_b = self.round_sample(sample_b)
    loss_reconstruction = self.imputation_reconstruction(data_expanded, None, sample_b)
    sample_b = self.sample_b_regularization(data_expanded, sample_b)
    data_imputed = self.imputation_function(data_expanded, sample_b)
    data_imputed = self.post_process(data_expanded, data_imputed, sample_b, index = index)
    return data_imputed, loss_reconstruction



# Example of simple imputation : 

class NoDestructionImputation(Imputation):
  
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,
             ):
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)

    
  def imputation_function(self, data_expanded, sample_b):
    return data_expanded

class SelectionAsInput(Imputation):
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,
             ):
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)


    
  def imputation_function(self, data_expanded, sample_b):
    return sample_b

  
class ConstantImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,):
    # self.cste = torch.tensor(cste).cuda()
    # self.cste = torch.tensor(cste) #TODO
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)

    self.cste = nn.parameter.Parameter(torch.tensor(cste), requires_grad=False)

  def has_constant(self):
    return True

  def get_constant(self):
    return self.cste

  def imputation_function(self, data_expanded, sample_b):
    result = data_expanded *  sample_b + (1-sample_b) * self.cste 
    return result



class NoiseImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,):
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)


  def imputation_function(self, data_expanded, sample_b):
    normal = torch.distributions.normal.Normal(torch.zeros_like(sample_b), torch.ones_like(sample_b))
    noise = normal.sample()
    return data_expanded * sample_b + (1-sample_b) *  noise 



class LearnConstantImputation(Imputation):
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None,):

    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)
    self.cste = nn.Parameter(torch.rand(1,), requires_grad = True)

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * sample_b + (1-sample_b) * self.cste 
   

  def has_constant(self):
    return True
  
  def get_constant(self):
    return self.cste
    
  def cuda(self):
    super().cuda()
    self.cste = torch.rand(1, requires_grad=True, device = "cuda")

  def zero_grad(self):
    super().zero_grad()
    if self.cste.grad is not None :
      self.cste.grad.zero_()


  def is_learnable(self):
    return True

class SumImpute(Imputation):
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = True, post_process_regularization = None,):
      super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                      sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization,)
      
  def imputation_function(self, data_expanded, sample_b):
      data_expanded_flatten = data_expanded.flatten(1)
      sample_b_flatten = sample_b.flatten(1)
      return torch.sum(data_expanded_flatten * sample_b_flatten, axis=1)
   

