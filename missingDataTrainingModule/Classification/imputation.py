import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

import torch

class Imputation():
  def __init__(self,input_size = (1,28,28), isRounded = False):
    self.isRounded = isRounded
    self.input_size = input_size
    self.kernel_updated = False
    
  def kernel_update(self, kernel_patch, stride_patch):
    self.kernel_updated = True
    assert(stride_patch[0]>0)
    assert(stride_patch[1]>0)
    assert(kernel_patch[0]>= stride_patch[0])
    assert(kernel_patch[1]>= stride_patch[1])
    self.kernel_patch = kernel_patch
    self.stride_patch = stride_patch
    self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(self.input_size, stride_patch)

  def round_sample(self, sample_b):
    if self.isRounded :
      sample_b_rounded = torch.round(sample_b)
      return sample_b_rounded #N_expectation, batch_size, channels, size:...
    else :
      return sample_b

  
  def patch_creation(self,sample_b):
    assert(self.kernel_updated)


    sample_b = sample_b.reshape((-1, self.input_size[0],self.nb_patch_x,self.nb_patch_y))
    if self.kernel_patch == (1,1):
      return sample_b
    else :
      batch_size = sample_b.shape[0]
      new_sample_b = torch.zeros((batch_size, self.input_size[0],self.input_size[1],self.input_size[2]))
      if sample_b.is_cuda :
        new_sample_b = new_sample_b.cuda()
      for channel in range(self.input_size[0]):

        for stride_idx in range(self.nb_patch_x):
          stride_x_location = stride_idx*self.stride_patch[0]
          remaining_size_x = min(self.input_size[1]-stride_x_location,self.kernel_patch[0])

          for stride_idy in range(self.nb_patch_y):
            stride_y_location = stride_idy*self.stride_patch[0]
            remaining_size_y = min(self.input_size[2]-stride_y_location,self.kernel_patch[1])
            
            # print(remaining_size_x)
            # print(remaining_size_y)
            # print(new_sample_b[:, channel, stride_x_location:stride_x_location+remaining_size_x, stride_y_location:stride_y_location+remaining_size_y].shape)
            # print(sample_b[:,channel, stride_idx, stride_idy].shape)
            # print(sample_b[:,channel, stride_idx, stride_idy].expand(-1, channel,remaining_size_x, remaining_size_y).shape)
            new_sample_b[:, channel, stride_x_location:stride_x_location+remaining_size_x, stride_y_location:stride_y_location+remaining_size_y] += \
              sample_b[:,channel, stride_idx, stride_idy].unsqueeze(1).unsqueeze(1).expand(-1, remaining_size_x, remaining_size_y)


      return new_sample_b
            

  def readable_sample(self, sample_b):
    return self.patch_creation(sample_b)

  def impute(self,data_expanded, sample_b):
    raise NotImplementedError

  def is_learnable(self):
    return False

  def add_channels(self):
    return False




class ConstantImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    super().__init__(input_size =input_size, isRounded = isRounded)
    
  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)

    return data_expanded * ((1-sample_b) * self.cste + sample_b)



class MaskConstantImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    super().__init__(input_size = input_size, isRounded = isRounded)

  def impute(self, data_expanded, sample_b):
  
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    return torch.cat([data_expanded * ((1-sample_b) * self.cste + sample_b), sample_b], axis = 2)

  def add_channels(self):
    return True

class LearnImputation(Imputation):
  def __init__(self, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.learned_cste = torch.rand(1, requires_grad=True)
    super().__init__(input_size = input_size, isRounded = isRounded)

  def get_learnable_parameter(self):
    return self.learned_cste

  def cuda(self):
    self.learned_cste = torch.rand(1, requires_grad=True, device = "cuda")

  def zero_grad(self):
    if self.learned_cste.grad is not None :
      self.learned_cste.grad.zero_()

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    return data_expanded  * ((1-sample_b) * self.learned_cste + sample_b)

  def is_learnable(self):
    return True

