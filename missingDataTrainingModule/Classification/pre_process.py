import torch
import torch.nn as nn
import numpy as np


##### PRE_PROCESS :

### SAMPLE_B REGULARIZATION :
class MaskRegularization(nn.Module):
  def __init__(self):
    super().__init__()

  def __call__(self, data_expanded, sample_b):
    raise NotImplementedError


class SimpleBRegularization(MaskRegularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    if self.rate > np.random.random():
      sample_b = torch.ones(data_expanded.shape).cuda()
    return sample_b


  
class LessDestructionRegularization(MaskRegularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    sample_b = torch.where(
      ((sample_b<0.5) * torch.rand(sample_b.shape, device = sample_b.device)>self.rate),
      torch.zeros(sample_b.shape,device = sample_b.device),
      sample_b
    )
    return sample_b


