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
            
  def __str__(self):
    string = str(type(self)).split(".")[-1].replace("'","").strip(">")
    if self.has_constant():
      string +="_"+ str(self.cste)
    if self.has_rate():
      string +=f"_rate_{self.rate}"
    return string

  def readable_sample(self, sample_b):
    return self.patch_creation(sample_b)

  def impute(self,data_expanded, sample_b):
    raise NotImplementedError

  def is_learnable(self):
    return False

  def has_constant(self):
    return False

  def has_rate(self):
    return False

  def add_channels(self):
    return False




class ConstantImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    super().__init__(input_size =input_size, isRounded = isRounded)

  def has_constant(self):
    return True


   

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)

    return data_expanded * ((1-sample_b) * self.cste + sample_b)


class ConstantImputationRateReg(Imputation):
  def __init__(self, rate = 0.5, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    self.rate = rate
    super().__init__(input_size =input_size, isRounded = isRounded)

  def has_constant(self):
    return True

  def has_rate(self):
    return True

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    if self.rate > np.random.random():
      return data_expanded

    return data_expanded * ((1-sample_b) * self.cste + sample_b)


class ConstantImputationInsideReg(Imputation):
  def __init__(self, rate = 0.5, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    self.rate = rate
    super().__init__(input_size =input_size, isRounded = isRounded)

  def has_constant(self):
    return True

  def has_rate(self):
    return True

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    
    sample_b = torch.where(
      ((sample_b<0.5) * torch.rand(sample_b.shape, device = "cuda")>self.rate),
      torch.zeros(sample_b.shape,device = "cuda"),
      sample_b
    )
    return data_expanded * ((1-sample_b) * self.cste + sample_b)

class ConstantImputationInsideReverseReg(Imputation):
  def __init__(self, rate = 0.5, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    self.rate = rate
    super().__init__(input_size =input_size, isRounded = isRounded)

  def has_constant(self):
    return True

  def has_rate(self):
    return True

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    
    sample_b = torch.where(
      (torch.rand(sample_b.shape, device = "cuda")>self.rate),
      1-sample_b,
      sample_b
    )
    return data_expanded * ((1-sample_b) * self.cste + sample_b)

class MaskConstantImputation(Imputation):
  def __init__(self, cste = 0, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = cste
    super().__init__(input_size = input_size, isRounded = isRounded)




  def has_constant(self):
    return True
  def impute(self, data_expanded, sample_b):
  
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    return torch.cat([data_expanded * ((1-sample_b) * self.cste + sample_b), sample_b], axis = 2)

  def add_channels(self):
    return True

class LearnConstantImputation(Imputation):
  def __init__(self, input_size = (1,28,28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    self.cste = torch.rand(1, requires_grad=True)
    super().__init__(input_size = input_size, isRounded = isRounded)

  def has_constant(self):
    return True

  def get_learnable_parameter(self):
    return self.cste

  def cuda(self):
    self.cste = torch.rand(1, requires_grad=True, device = "cuda")

  def zero_grad(self):
    if self.cste.grad is not None :
      self.cste.grad.zero_()

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    return data_expanded  * ((1-sample_b) * self.cste + sample_b)

  def is_learnable(self):
    return True

class PermutationInvariance(Imputation):

  def __init__(self, h_network, size_D = 10, add_index = True, input_size = (1, 28, 28), kernel_patch = (1,1), stride = (1,1), isRounded = False):
    super().__init__(input_size = input_size, isRounded = isRounded)
    self.channel_output = size_D + 1


    self.h_network = h_network
    # self.embedding = torch.nn.embedding
    # self.e_m = torch.empty((size_D,np.prod(input_size)),requires_grad = True) # e_m can be considered as embedding withotu mask select
    # torch.nn.init.normal_(self.e_m)
    # self.e_m = self.e_m.double()
    # self.e_m = 
    self.add_index = add_index
    self.size_D = size_D

    self.e_m = torch.rand((self.size_D,np.prod(self.input_size)), requires_grad=True)

    if self.add_index :
      aux = []
      for channel in range(input_size[0]):
        for index_x in range(input_size[1]):
          for index_y in range(input_size[2]):
            aux.append(channel+index_x+index_y)
      self.index = torch.tensor(aux).unsqueeze(0)
      self.channel_output+=1


    self.identity = torch.ones((1,input_size[0]*input_size[1]*input_size[2]))

  
  def is_learnable(self):
    return True


  def get_learnable_parameter(self):
    parameter = []
    # for embedding in self.embedding :
      # parameter.append({"params": embedding.parameters()})
    # parameter.append({"params": self.h_network.parameters()})
    parameter.append({"params": self.e_m})
    return parameter
    

  def train(self):
      self.h_network.train()
      self.e_m.requires_grad_(True)

  def cuda(self):
      self.h_network = self.h_network.cuda()
      # self.e_m = torch.empty((self.size_D,np.prod(self.input_size)),requires_grad = True) # e_m can be considered as embedding withotu mask select
      # torch.nn.init.normal_(self.e_m)
      self.e_m = torch.rand((self.size_D,np.prod(self.input_size)), requires_grad=True, device = "cuda")
      # self.e_m = self.e_m.double()
      # self.e_m = self.e_m.cuda()


      if self.add_index :
        self.index = self.index.cuda()

  def eval(self):
      self.h_network.eval()
      self.e_m.requires_grad_(False)

  def zero_grad(self):
      self.h_network.zero_grad()
      if self.e_m.grad is not None :
        self.e_m.grad.zero_()

  def impute(self, data_expanded, sample_b):
    # How should I inpute with this if it's not integer ?
    batch_size = data_expanded.shape[0]
    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)
    sample_b = sample_b.flatten(1)
    data_expanded_flatten = data_expanded.flatten(1)
    # print(sample_b.shape)
    # new_data = torch.masked_select(data_expanded_flatten, sample_b<0.5).reshape(batch_size,-1)
    # print(self.e_m.shape)
    # print(sample_b.shape)
    # print(self.e_m.unsqueeze(0).expand((batch_size,-1,-1)).shape)
    # print((sample_b.unsqueeze(1).expand((-1,self.size_D,-1))<0.5).shape)
    complete_output = []
    for i in range(batch_size):
      new_data = torch.masked_select(data_expanded_flatten[i], sample_b[i]<0.5)

      # e_m = torch.masked_select(self.e_m.unsqueeze(0).expand((batch_size,-1,-1)),
                                # sample_b.unsqueeze(1).expand((-1,self.size_D,-1))<0.5)
      # print(self.e_m.shape)
      e_m = torch.masked_select(self.e_m,
                              sample_b[i].unsqueeze(0).expand((self.size_D,-1))<0.5).reshape(self.size_D,-1)
                                                          
      # output = torch.tensordot(e_m, new_data, dims = (-1,-1))

      # print(new_data.shape)
      # print(new_data.unsqueeze(1).expand((-1, self.size_D, -1)).shape)
      # print(e_m.shape)
      # print(new_data.shape)
      output_em = e_m * new_data.unsqueeze(0).expand(( self.size_D, -1))

      output_identity = new_data.unsqueeze(0)
      # print(output_identity.shape)
      # print(output_em.shape)
      output = torch.cat([output_em, output_identity], dim = 0)
      # print(self.index.shape)
      if self.add_index :
        index = torch.masked_select(self.index, sample_b[i].unsqueeze(0)<0.5)
        output_index = index * new_data.unsqueeze(0)
        output = torch.cat([output, output_index], dim = 0)
      output = torch.transpose(output, 0 ,1)
      
      # print(output.shape)
    
      output = torch.sum(self.h_network(output),dim = 0)    
      complete_output.append(output.unsqueeze(0))

    imputed_data = torch.cat(complete_output,dim=0)
    return imputed_data