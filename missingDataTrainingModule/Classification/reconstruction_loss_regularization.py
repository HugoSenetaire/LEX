import torch
import torch.nn as nn




##### Multiple Imputation
  

class NetworkBasedReconstructionRegularization(nn.Module):
  def __init__(self, network_reconstruction, lambda_reconstruction=0.1,):
    super(NetworkBasedReconstructionRegularization, self).__init__()
    self.network_reconstruction = network_reconstruction
    self.lambda_reconstruction = nn.Parameter(torch.tensor(lambda_reconstruction, dtype = torch.float32), requires_grad = False)

  def __call__(self, data_imputed, data, mask, index = None):
    raise NotImplementedError



### LOSS REGULARIZATION : 

class AutoEncoderReconstructionRegularization(NetworkBasedReconstructionRegularization):
  def __init__(self, network_reconstruction, lambda_reconstruction = 0.1,):
    super().__init__(network_reconstruction = network_reconstruction, lambda_reconstruction = lambda_reconstruction,)

  
  def __call__(self, data_imputed, data, mask, index = None,):
    data_reconstruced = self.network_reconstruction(data_imputed)
    loss = self.lambda_reconstruction * nn.functional.mse_loss(data_reconstruced, data)
    return loss

