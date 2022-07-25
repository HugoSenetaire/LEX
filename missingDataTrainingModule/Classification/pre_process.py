import torch
import torch.nn as nn
import numpy as np


##### PRE_PROCESS :

### MASK REGULARIZATION :
class MaskRegularization(nn.Module):
  def __init__(self):
    super().__init__()

  def __call__(self, data, mask):
    raise NotImplementedError


class RateMaskRegularization(MaskRegularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data, mask):
    if self.rate > np.random.random():
      mask = torch.ones(data.shape).cuda()
    return mask


  
class RateMaskRegularization2(MaskRegularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data, mask):
    mask = torch.where(
      ((mask<0.5) * torch.rand(mask.shape, device = mask.device)>self.rate),
      torch.zeros(mask.shape,device = mask.device),
      mask
    )
    return mask



list_maskreg = {
  "None" : None,
  "rate_mask" : RateMaskRegularization,
  "rate_mask2" : RateMaskRegularization2,
}

def get_maskreg(mask_reg):
  if mask_reg == "None" or mask_reg== None :
    return None
  elif mask_reg in list_maskreg :
    return list_maskreg[mask_reg]
  else :
    raise ValueError("mask_reg must be None, 'rate_mask' or 'rate_mask2'")

