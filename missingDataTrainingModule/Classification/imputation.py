import sys
import os
from .post_process_imputation import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
import torch


def prepare_process(input_process):
  """ Utility function to make sure that the input_process is a list of function or None"""
  if input_process is not list and input_process is not None :
    input_process = [input_process]
  return input_process

class Imputation():
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.isRounded = isRounded
    self.input_size = input_size
    self.kernel_updated = False
    self.add_mask = add_mask
   
    self.reconstruction_reg = prepare_process(reconstruction_reg)
    self.post_process_regularization = prepare_process(post_process_regularization)
    self.sample_b_reg = prepare_process(sample_b_reg)

    
    self.nb_imputation = 1
    if self.post_process_regularization is not None :
      for element in self.post_process_regularization :
        if element.multiple_imputation :
          self.nb_imputation *= element.nb_imputation
          

    if self.add_mask :
      self.add_channels = True


  def is_learnable(self):
    """ Check if any process for the imputation needs training """
    
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          return True

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          return True

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          return True

    return False

  def get_learnable_parameter(self):
    """ Check every element in the imputation method and add it to the parameters_list if it needs training"""

    parameter =[]
    network_list = []

    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          network_list.append(process.network)
          parameter.append({"params":process.parameters()})

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          parameter.append({"params":process.parameters()})
    
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
            if process.network not in network_list :
              network_list.append(process.network)
              parameter.append({"params":process.parameters()})

    return parameter


  def cuda(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          process.cuda()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          process.cuda()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          process.cuda()


    

  def round_sample(self, sample_b):
    """ Round sample so that we only have a discrete mask. Unadvised as the training becomes very unstable"""
    if self.isRounded :
      sample_b_rounded = torch.round(sample_b)
      return sample_b_rounded #N_expectation, batch_size, channels, size:...
    else :
      return sample_b

  
            
  def __str__(self):
    """ Make sure of what is printed when everything is in it """
    string = str(type(self)).split(".")[-1].replace("'","").strip(">")
    if self.has_constant():
      string +="_"+ str(self.cste)
    if self.has_rate():
      string +=f"_rate_{self.rate}"
    return string

  def imputation_reconstruction(self, data_expanded, data_imputed, sample_b):
    loss_reconstruction = torch.zeros((1)).cuda()
    if self.reconstruction_reg is not None :
      data_imputed = self.imputation_function(data_expanded, sample_b)
      for process in self.reconstruction_reg :
          loss_reconstruction += process(data_expanded, data_imputed, sample_b)
    return loss_reconstruction

  
  def add_mask_method(self, data_imputed, sample_b):
    if len(sample_b.shape)>2:
      sample_b_aux = sample_b[:,0].unsqueeze(1)
    else :
      sample_b_aux = sample_b
    return torch.cat([data_imputed, sample_b_aux], axis =1)


  def post_process(self, data_expanded, data_imputed, sample_b):
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        data_imputed, data_expanded, sample_b = process(data_expanded, data_imputed, sample_b)
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

  def impute(self,data_expanded, sample_b):

    sample_b = self.round_sample(sample_b)
    loss_reconstruction = self.imputation_reconstruction(data_expanded, None, sample_b)
    sample_b = self.sample_b_regularization(data_expanded, sample_b)
    data_imputed = self.imputation_function(data_expanded, sample_b)

    data_imputed = self.post_process(data_expanded, data_imputed, sample_b)

    return data_imputed, loss_reconstruction

  def eval(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        process.eval()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        process.eval()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        process.eval()

  def train(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        process.train()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        process.train()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        process.train()

  def zero_grad(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          process.zero_grad()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          process.zero_grad()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          process.zero_grad()

    
    

  def has_constant(self):
    return False

  def has_rate(self):
    return False



# Example of simple imputation : 

class ConstantImputation(Imputation):

 
  def __init__(self, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def has_constant(self):
    return True

  def get_constant(self):
    return self.cste

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)


class NoiseImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def has_constant(self):
    return True

  def get_constant(self):
    return self.cste

  def imputation_function(self, data_expanded, sample_b):
    normal = torch.distributions.normal.Normal(torch.zeros(sample_b.shape), torch.ones(sample_b.shape))
    noise = normal.sample().cuda()
    return data_expanded * sample_b + (1-sample_b) *  noise 



class LearnConstantImputation(Imputation):
  def __init__(self, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.rand(1, requires_grad=True)
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)
   

  def has_constant(self):
    return True
  
  def get_constant(self):
    return self.cste
    

  def get_learnable_parameter(self):
    parameters = super().get_learnable_parameter()
    parameters.append({"params":self.cste})
    return parameters

  def train(self):
    super().train()
    self.cste.requires_grad_(True)
  
  def eval(self):
    super().eval()
    self.cste.requires_grad_(False)

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
               add_mask = True, post_process_regularization = None):
      self.cste = torch.tensor(cste).cuda()
      add_mask = True
      super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                      sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)
      
  def imputation_function(self, data_expanded, sample_b):
      data_expanded_flatten = data_expanded.flatten(1)
      sample_b_flatten = sample_b.flatten(1)
      return torch.sum(data_expanded_flatten * sample_b_flatten, axis=1)
   

