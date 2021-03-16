import sys
import os
from .utils_imputation import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

import torch

class Imputation():
  def __init__(self,input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.isRounded = isRounded
    self.input_size = input_size
    self.kernel_updated = False
    if reconstruction_reg is not list and reconstruction_reg is not None :
      reconstruction_reg = [reconstruction_reg]
    self.reconstruction_reg = reconstruction_reg
    self.add_mask = add_mask

    if post_process_regularization is not list and post_process_regularization is not None :
      post_process_regularization = [post_process_regularization]
    self.post_process_regularization = post_process_regularization
    

    if sample_b_reg is not list and sample_b_reg is not None :
      print(sample_b_reg)
      sample_b_reg = [sample_b_reg]
    print(sample_b_reg)
    self.sample_b_reg = sample_b_reg

    if self.add_mask :
      self.add_channels = True


  def is_learnable(self):
    
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

    parameter =[]
    network_list = []

    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          # parameter.append(process.parameters())
          # print(process.network)
          network_list.append(process.network)
          parameter.append({"params":process.parameters()})

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          # parameter.append(process.parameters())
          parameter.append({"params":process.parameters()})
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          # parameter.append(process.parameters())
            if process.network not in network_list :
              network_list.append(process.network)
              # print(process.network)
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



    
  def kernel_update(self, kernel_patch, stride_patch):
    
    self.kernel_updated = True
    if self.input_size is int or len(self.input_size)<=2:
      self.image = False
      self.kernel_patch = (1,1)
      self.stride_patch = (1,1)
      try :
        self.nb_patch_x, self.nb_patch_y = int(self.input_size), 1
      except :
        self.nb_patch_x, self.nb_patch_y = int(self.input_size[0]), int(self.input_size[1])
    else :
      self.image = True
      assert(kernel_patch[0]>= stride_patch[0])
      assert(kernel_patch[1]>= stride_patch[1])
      assert(stride_patch[0]>0)
      assert(stride_patch[1]>0)
      self.kernel_patch = kernel_patch
      self.stride_patch = stride_patch
      self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(self.input_size, self.stride_patch)

  def round_sample(self, sample_b):
    if self.isRounded :
      sample_b_rounded = torch.round(sample_b)
      return sample_b_rounded #N_expectation, batch_size, channels, size:...
    else :
      return sample_b

  
  def patch_creation(self,sample_b):
    assert(self.kernel_updated)
    
    if self.image :

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
              
         
              new_sample_b[:, channel, stride_x_location:stride_x_location+remaining_size_x, stride_y_location:stride_y_location+remaining_size_y] += \
                sample_b[:,channel, stride_idx, stride_idy].unsqueeze(1).unsqueeze(1).expand(-1, remaining_size_x, remaining_size_y)
      
    else :
      new_sample_b = sample_b
    return new_sample_b
            
  def __str__(self):
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
          print(process)
          loss_reconstruction += process(data_expanded, data_imputed, sample_b)

      

    return loss_reconstruction



  

  def sample_b_regularization(self, data_expanded, sample_b):
    if self.sample_b_reg is None :
      return sample_b
    else :
      for process in self.sample_b_reg :
        sample_b = process(data_expanded, sample_b)
      

      return sample_b

  def post_process(self, data_expanded, data_imputed, sample_b):
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        data_imputed = process(data_expanded, data_imputed, sample_b)
    if self.add_mask:
      data_imputed = torch.cat([data_imputed, sample_b], axis = 1)
      
    return data_imputed

  def readable_sample(self, sample_b):
    return self.patch_creation(sample_b)

  def imputation_function(self, data_expanded, sample_b):
    raise NotImplementedError

  def impute(self,data_expanded, sample_b):

    sample_b = self.round_sample(sample_b)
    sample_b = self.patch_creation(sample_b)

    loss_reconstruction = self.imputation_reconstruction(data_expanded, None, sample_b)
    sample_b = self.sample_b_regularization(data_expanded, sample_b)
    data_imputed = self.imputation_function(data_expanded, sample_b)
    data_imputed = self.post_process(data_expanded, data_imputed, sample_b)

    return data_imputed, loss_reconstruction

  def eval(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          process.eval()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          process.eval()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          process.eval()

  def train(self):
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg:
        if process.to_train :
          process.train()

    if self.sample_b_reg is not None :
      for process in self.sample_b_reg:
        if process.to_train :
          process.train()

    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        if process.to_train:
          process.train()
    

  def has_constant(self):
    return False

  

  def has_rate(self):
    return False





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

class ConstantImputationRateReg(ConstantImputation):
  def __init__(self, rate = 0.5, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    self.rate = rate
    sample_b_reg = SimpleB_Regularization(rate = self.rate)
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def has_constant(self):
    return True

  def has_rate(self):
    return True

  
  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)


class ConstantImputationInsideReg(ConstantImputation):
  def __init__(self, rate = 0.5, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    self.rate = rate

    sample_b_reg =  Less_Destruction_Regularization(rate = self.rate)
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)
   

  def has_constant(self):
    return True

  def has_rate(self):
    return True

class ConstantImputationInsideReverseReg(ConstantImputation):
  def __init__(self,cste = 0, rate = 0.5, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    self.rate = rate

    sample_b_reg = Complete_Inversion_Regularization(rate = self.rate)
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)
   

  def has_constant(self):
    return True

  def has_rate(self):
    return True


class MaskConstantImputation(ConstantImputation):
  def __init__(self, cste = 0, input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = True, post_process_regularization = None):
    self.cste = torch.tensor(cste).cuda()
    add_mask = True
    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)

  def imputation_function(self, data_expanded, sample_b):
    return data_expanded * ((1-sample_b) * self.cste + sample_b)
   


  def has_constant(self):
    return True



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

  # def impute(self, data_expanded, sample_b):
  #   sample_b = self.round_sample(sample_b)
  #   sample_b = self.patch_creation(sample_b)
  #   return data_expanded  * ((1-sample_b) * self.cste + sample_b)

  def is_learnable(self):
    return True


# class AutoEncoderImputation(Imputation):
#   def __init__(self, autoencoder, to_train = True,
#                input_size = (1,28,28), isRounded = False,
#                reconstruction_reg = None, sample_b_reg = None,
#                add_mask = False, post_process_regularization = None):


#     super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
#                     sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)
#     self.autoencoder = autoencoder
#     self.cste = 0
#     self.to_train = to_train
#     # if self.to_train :
#     #   self.autoencoder.requires_grad(False)
#     # else :
#     #   self.autoencoder.requires_grad(True)
#     if not self.to_train:
#       for param in self.autoencoder.parameters():
#         param.requires_grad = False

#   def imputation_function(self, data_expanded, sample_b):
#     new_data = data_expanded  * ((1-sample_b) * self.cste + sample_b)
#     data_imputed = self.autoencoder(new_data)
#     return data_imputed
   


#   def train(self):
#     if self.to_train :
#       self.autoencoder.train()

#   def eval(self):
#     self.autoencoder.eval()


# class CombinedAutoEncoderImputation(Imputation):
#   def __init__(self, autoencoder, to_train = True,
#                input_size = (1,28,28), isRounded = False,
#                reconstruction_reg = None, sample_b_reg = None,
#                add_mask = False, post_process_regularization = None):


#     super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
#                     sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)
#     self.autoencoder = autoencoder
#     self.cste = 0
#     self.to_train = to_train
#     # if self.to_train :
#     #   self.autoencoder.requires_grad(False)
#     # else :
#     #   self.autoencoder.requires_grad(True)
#     if not self.to_train:
#       for param in self.autoencoder.parameters():
#         param.requires_grad = False

#   def imputation_function(self, data_expanded, sample_b):
#     data_imputed = data_expanded  * ((1-sample_b) * self.cste + sample_b)
#     return data_imputed
   

#   def is_learnable(self):
#     if self.to_train :
#       return True
#     else :
#       return False

#   def get_learnable_parameter(self):

#     parameter =[]
#     parameter.append({"params":self.autoencoder.parameters()})
#     return parameter


#   def cuda(self):
#     self.autoencoder = self.autoencoder.cuda()

#   def zero_grad(self):
#     if self.to_train :
#       self.autoencoder.zero_grad()

#   def train(self):
#     if self.to_train :
#       self.autoencoder.train()

#   def eval(self):
#     self.autoencoder.eval()





class PermutationInvariance(Imputation):

  def __init__(self, h_network, size_D = 10, add_index = True, 
               input_size = (1,28,28), isRounded = False,
               reconstruction_reg = None, sample_b_reg = None,
               add_mask = False, post_process_regularization = None):


    super().__init__(input_size =input_size, isRounded = isRounded, reconstruction_reg=reconstruction_reg,
                    sample_b_reg=sample_b_reg, add_mask= add_mask, post_process_regularization=post_process_regularization)
    self.channel_output = size_D + 1


    self.h_network = h_network 
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
    parameter.append({"params": self.e_m})
    return parameter
    

  def train(self):
      self.h_network.train()
      self.e_m.requires_grad_(True)

  def cuda(self):
      self.h_network = self.h_network.cuda()
      self.e_m = torch.rand((self.size_D,np.prod(self.input_size)), requires_grad=True, device = "cuda")



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

    complete_output = []
    for i in range(batch_size):
      new_data = torch.masked_select(data_expanded_flatten[i], sample_b[i]<0.5)

      e_m = torch.masked_select(self.e_m,
                              sample_b[i].unsqueeze(0).expand((self.size_D,-1))<0.5).reshape(self.size_D,-1)
                                                          
      output_em = e_m * new_data.unsqueeze(0).expand(( self.size_D, -1))

      output_identity = new_data.unsqueeze(0)
      output = torch.cat([output_em, output_identity], dim = 0)

      if self.add_index :
        index = torch.masked_select(self.index, sample_b[i].unsqueeze(0)<0.5)
        output_index = index * new_data.unsqueeze(0)
        output = torch.cat([output, output_index], dim = 0)
      output = torch.transpose(output, 0 ,1)

      output = torch.sum(self.h_network(output),dim = 0)    
      complete_output.append(output.unsqueeze(0))

    imputed_data = torch.cat(complete_output,dim=0)
    return imputed_data