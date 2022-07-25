import sys
import os

import torch
import torch.nn as nn


## Different Utils :
def prepare_process(input_process):
  """ Utility function to make sure that the input_process is a list of function or None"""
  if input_process is None:
    return None
  else :
    if input_process is not list :
      input_process = [input_process]
    input_process = nn.ModuleList(input_process)

    return input_process





def expand_for_imputations(data, mask, nb_imputation_iwae, nb_imputation_mc, index = None, collapse = False):
    wanted_reshape = torch.Size((1,)) + torch.Size((data.shape[0],)) + torch.Size((1,)) + data.shape[1:]
    wanted_transform = torch.Size((nb_imputation_mc,)) + torch.Size((data.shape[0],)) + torch.Size((nb_imputation_iwae,)) + data.shape[1:]
    data_expanded = data.reshape(wanted_reshape).expand(wanted_transform)
    mask_expanded = mask.reshape(wanted_reshape).expand(wanted_transform)
    if index is not None :
      wanted_reshape = torch.Size((1,)) + torch.Size((index.shape[0],)) + torch.Size((1,)) + index.shape[1:]
      wanted_transform_index = torch.Size((nb_imputation_iwae,)) + index.shape + torch.Size((nb_imputation_mc,))
      index_expanded = index.reshape(wanted_reshape).expand(wanted_transform_index)
    else:
      index_expanded = None

    if collapse :
      data_expanded = data_expanded.flatten(0, 2)
      mask_expanded = mask_expanded.flatten(0, 2)
      if index is not None :
        index_expanded = index_expanded.flatten(0, 2)
    return data_expanded, mask_expanded, index_expanded


## Abstract imputation class :

class Imputation(nn.Module):
  def __init__(self, nb_imputation_iwae = 1,
            nb_imputation_mc = 1,
            nb_imputation_iwae_test = None,
            nb_imputation_mc_test = None,
            reconstruction_reg = None,
            mask_reg = None,
            add_mask = False,
            post_process_regularization = None,
            **kwargs):
    super().__init__()
    self.add_mask = add_mask
    self.reconstruction_reg = prepare_process(reconstruction_reg)
    self.post_process_regularization = prepare_process(post_process_regularization)
    self.mask_reg = prepare_process(mask_reg)

    self.nb_imputation_iwae = nb_imputation_iwae
    if nb_imputation_iwae_test is None :
      self.nb_imputation_iwae_test = 1
    else :
      self.nb_imputation_iwae_test = nb_imputation_iwae_test

    self.nb_imputation_mc = nb_imputation_mc
    if nb_imputation_mc_test is None :
      self.nb_imputation_mc_test = 1
    else :
      self.nb_imputation_mc_test = nb_imputation_mc_test
  
  def has_constant(self):
    return False
 
  def has_rate(self):
    return False



  def reconstruction_regularization(self, data_imputed, data, mask, index = None):
    loss_reconstruction = torch.tensor(0., device = data.device)
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg :
          loss_reconstruction += process(data_imputed, data, mask, index = index)
    return loss_reconstruction

  
  def add_mask_method(self, data_imputed, mask):
    if len(mask.shape)>2:
      mask_aux = mask[:,0].unsqueeze(1)
    else :
      mask_aux = mask
    return torch.cat([data_imputed, mask_aux], axis =1)


  def post_process(self, data_imputed, data, mask, index = None):
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        data_imputed, mask = process(data_imputed, data, mask, index = index)
    if self.add_mask:
      data_imputed = self.add_mask_method(data_imputed, mask)
    return data_imputed

  def mask_regularization(self, data, mask):
    if self.mask_reg is None :
      return mask
    else :
      for process in self.mask_reg :
        mask = process(data, mask)
      return mask


  def imputation_function(self, data, mask, index=None):
    raise NotImplementedError

  def forward(self, data, mask, index = None):
    mask = self.mask_regularization(data, mask)

    if self.training :
      nb_imputation_mc = self.nb_imputation_mc
      nb_imputation_iwae = self.nb_imputation_iwae
    else :
      nb_imputation_mc = self.nb_imputation_mc_test
      nb_imputation_iwae = self.nb_imputation_iwae_test

    data_expanded, mask_expanded, index_expanded = expand_for_imputations(data, mask, nb_imputation_mc = nb_imputation_mc, nb_imputation_iwae=nb_imputation_iwae, index = index, collapse = True)
    data_imputed = self.imputation_function(data_expanded, mask_expanded, index_expanded)
    loss_reconstruction = self.reconstruction_regularization(data_imputed, data_expanded, mask_expanded, index = index_expanded)
    data_imputed = self.post_process(data_imputed, data_expanded, mask_expanded, index = index_expanded)
    return data_imputed, loss_reconstruction



# Example of simple imputation : 

class NoDestructionImputation(Imputation):  
  def __init__(self, 
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
               ):
    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    **kwargs,
                  )

  def imputation_function(self, data, mask, index = None):
    return data



class MaskAsInput(Imputation):
  def __init__(self,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
             ):
    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                  reconstruction_reg=reconstruction_reg,
                  mask_reg=mask_reg,
                  add_mask= add_mask,
                  post_process_regularization=post_process_regularization,
                  )
    
  def imputation_function(self, data, mask, index = None):
    return mask

class ConstantImputation(Imputation):
  def __init__(self,
              cste = 0,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
              ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    )

    self.cste = nn.parameter.Parameter(torch.tensor(cste), requires_grad=False)

  def has_constant(self):
    return True

  def get_constant(self):
    return self.cste

  def imputation_function(self, data, mask, index = None):
    data_imputed = data *  mask + (1-mask) * self.cste 
    return data_imputed

class MultipleConstantImputation(Imputation):
  def __init__(self, 
              cste_list_dim = [-2, 2],
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
            ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    )

    self.cste_list_dim = nn.parameter.Parameter(torch.tensor(cste_list_dim), requires_grad=False)

  def has_constant(self):
    return False


  def imputation_function(self, data, mask, index = None):
    data_imputed = data *  mask + (1-mask) * self.cste_list_dim
    return data_imputed


class NoiseImputation(Imputation):
  def __init__(self,
              sigma = 1.0,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
              ):
    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                  reconstruction_reg=reconstruction_reg,
                  mask_reg=mask_reg,
                  add_mask= add_mask,
                  post_process_regularization=post_process_regularization,
                  )
    self.sigma = sigma
    assert self.sigma > 0

  def imputation_function(self, data, mask, index = None):
    normal = torch.distributions.normal.Normal(torch.zeros_like(mask), torch.full_like(mask, fill_value= self.sigma))
    noise = normal.sample()
    return data * mask + (1-mask) *  noise 



class LearnConstantImputation(Imputation):
  def __init__(self, 
              cste=None,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
              ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    **kwargs,
                    )

    if cste is None :
      cste = torch.rand(1,).type(torch.float32)
    self.cste = nn.Parameter(cste, requires_grad = True)

  def imputation_function(self, data, mask, index = None):
    return data * mask + (1-mask) * self.cste 
   


class SumImpute(Imputation):
  def __init__(self,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = True,
              post_process_regularization = None,
              **kwargs):
      super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                      nb_imputation_mc = nb_imputation_mc,
                      nb_imputation_iwae_test = nb_imputation_iwae_test,
                      nb_imputation_mc_test = nb_imputation_mc_test, 
                      reconstruction_reg=reconstruction_reg, 
                      mask_reg=mask_reg,
                      add_mask= add_mask,
                      post_process_regularization=post_process_regularization,)
      
  def imputation_function(self, data, mask, index = None):
      return torch.sum(data * mask, axis=-1)
   

class ModuleImputation(Imputation):
  def __init__(self, 
              module,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None, 
              add_mask = True,
              post_process_regularization = None,
              **kwargs):
      super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                      nb_imputation_mc = nb_imputation_mc,
                      nb_imputation_iwae_test = nb_imputation_iwae_test,
                      nb_imputation_mc_test = nb_imputation_mc_test,
                      reconstruction_reg=reconstruction_reg, 
                      mask_reg=mask_reg,
                      add_mask= add_mask,
                      post_process_regularization=post_process_regularization,)
      self.module = module
      
  def imputation_function(self, data, mask, index = None):
      imputation = self.module(data, mask, index = index)
      data_imputed = data * mask + (1-mask) * imputation
      return data_imputed


class DatasetBasedImputation(Imputation):
    def __init__(self, 
                dataset,
                nb_imputation_iwae = 1,
                nb_imputation_mc = 1,
                nb_imputation_iwae_test = None,
                nb_imputation_mc_test = None,
                reconstruction_reg = None,
                mask_reg = None, 
                add_mask = True,
                post_process_regularization = None,
                **kwargs):
        super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                        nb_imputation_mc = nb_imputation_mc,
                        nb_imputation_iwae_test = nb_imputation_iwae_test,
                        nb_imputation_mc_test = nb_imputation_mc_test,
                        reconstruction_reg=reconstruction_reg, 
                        mask_reg=mask_reg,
                        add_mask= add_mask,
                        post_process_regularization=post_process_regularization,)
        self.dataset = dataset
        self.exist = hasattr(dataset, "impute") 
        if not self.exist :
          self.nb_imputation = 1
          print(f"There is no theoretical method for multiple imputation with {dataset}. DatasetBasedImputation is bypassed from now on.")
        

    def imputation_function(self, data, mask, index = None):
        if self.exist :
          if self.training :
            dataset_type = "Train"
          else :
            dataset_type = "Test"
          imputed_output = self.dataset.impute(value = data.detach(), mask = mask.detach(), index = index, dataset_type = dataset_type)
          data_imputed = mask * data + (1-mask) * imputed_output
          return data_imputed
        else :
          return data



imputation_list = {
  "LearnConstantImputation" : LearnConstantImputation,
  "SumImpute" : SumImpute,
  "ModuleImputation" : ModuleImputation,
  "DatasetBasedImputation" : DatasetBasedImputation,
  "NoiseImputation" : NoiseImputation,
  "ConstantImputation" : ConstantImputation,
  "MultipleConstantImputation" : MultipleConstantImputation,
  "MaskAsInput" : MaskAsInput,
  "NoDestructionImputation" : NoDestructionImputation,  


}

## Imputation :
def get_imputation_type(imputation_name):  
  if imputation_name in imputation_list :
    return imputation_list[imputation_name]
  else :
    raise ValueError(f"Imputation {imputation_name} is not implemented")