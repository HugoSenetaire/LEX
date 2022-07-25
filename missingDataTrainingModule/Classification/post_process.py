
import torch
import torch.nn as nn



##### Post Process Abstract
  

class PostProcess(nn.Module):
  def __init__(self, network_post_process, trainable = False, **kwargs):
    super(PostProcess, self).__init__()
    self.network_post_process = network_post_process
    self.trainable = False

    if self.network_post_process is not None :
      for param in self.network_post_process.parameters():
        param.requires_grad = trainable


  def __call__(self, data_imputed, data, mask,index = None):
    raise NotImplementedError


  
### POST PROCESS REGULARIZATION :


class addNoiseToUnmask(PostProcess):
  def __init__(self, sigma_noise = 0.1, trainable = False, **kwargs):
    super(addNoiseToUnmask, self).__init__(network_post_process= None, trainable = trainable,)
    self.sigma_noise = sigma_noise

  def __call__(self, data_imputed, data, mask,index = None):
    data_imputed_noisy = data_imputed + self.sigma_noise * torch.randn_like(data_imputed) * mask
    return data_imputed_noisy, mask

class NetworkTransform(PostProcess):
  def __init__(self, network_post_process, trainable = False, **kwargs):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)

  def __call__(self, data_imputed, data, mask,index = None,):
    data_reconstructed = self.network_post_process(data_imputed)
    return data_reconstructed, mask
  

class NetworkAdd(PostProcess):
  def __init__(self, network_post_process, trainable = False, **kwargs):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)


  def __call__(self, data_imputed, data, mask, index = None,):
    data_reconstructed = self.network_post_process(data_imputed)
    data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
    return data_imputed, mask
  


class NetworkTransformMask(PostProcess):
  def __init__(self, network_post_process, trainable = False, **kwargs):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)

  def __call__(self, data_imputed, data, mask,index = None,):
    data_reconstructed = data_imputed * (1-mask) + self.network_post_process(data_imputed) * mask 
    return data_reconstructed, mask



list_post_process_reg = {
  "None" : None,
  "addNoiseToUnmask" : addNoiseToUnmask,
  "NetworkTransform" : NetworkTransform,
  "NetworkAdd" : NetworkAdd,
  "NetworkTransformMask" : NetworkTransformMask,
}

def get_post_process_reg(post_process_regularization):
  if post_process_regularization == None or post_process_regularization == "None" :
    return None
  elif post_process_regularization in list_post_process_reg :
    return list_post_process_reg[post_process_regularization]
  else :
    raise ValueError("Post process regularization not found")