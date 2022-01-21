
import torch
import torch.nn as nn



##### Post Process Abstract
  

class NetworkBasedPostProcess(nn.Module):
  def __init__(self, network_post_process, trainable = False,):
    super(NetworkBasedPostProcess, self).__init__()
    self.network_post_process = network_post_process
    self.trainable = False


    for param in self.network_post_process.parameters():
      param.requires_grad = trainable


  def __call__(self, data_expanded, data_imputed, sample_b,index = None):
    raise NotImplementedError


  
### POST PROCESS REGULARIZATION :

class NetworkTransform(NetworkBasedPostProcess):
  def __init__(self, network_post_process, trainable = False, ):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    data_reconstructed = self.network_post_process(data_imputed)
    return data_reconstructed, data_expanded, sample_b
  

class NetworkAdd(NetworkBasedPostProcess):
  def __init__(self, network_post_process, trainable = False, ):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)


  def __call__(self, data_expanded, data_imputed, sample_b, index = None,):
    data_reconstructed = self.network_post_process(data_imputed)
    data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
    return data_reconstructed, data_expanded, sample_b
  


class NetworkTransformMask(NetworkBasedPostProcess):
  def __init__(self, network_post_process, trainable = False, ):
    super().__init__(network_post_process = network_post_process, trainable = trainable,)

  def __call__(self, data_expanded, data_imputed, sample_b,index = None,):
    data_reconstructed = data_imputed * (1-sample_b) + self.network_post_process(data_imputed) * sample_b 
    return data_reconstructed, data_expanded, sample_b


