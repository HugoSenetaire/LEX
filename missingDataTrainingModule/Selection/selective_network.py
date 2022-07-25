import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from .utils_UNET import *

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np




class AbstractSelector(nn.Module):
  def __init__(self,input_size = (1,28,28), output_size = (1,28,28)):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_size
    if self.output_size is None:
      self.output_size = self.input_size
    
  def __call__(self, x):
    raise NotImplementedError


class SelectorLinear(AbstractSelector):
    def __init__(self,input_size = (1,28,28), output_size = (1,28,28)):
      super().__init__(input_size = input_size, output_size = output_size)
      self.pi = nn.Linear(np.prod(self.input_size), np.prod(self.output_size))
      

    def __call__(self, x):
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        return self.pi(x)

class SelectorLVL1(AbstractSelector):
    def __init__(self,input_size = (1,28,28), output_size = (1,28,28)):
      super().__init__(input_size = input_size, output_size = output_size)

      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.pi = nn.Linear(50, np.prod(self.output_size))
      

    def __call__(self, x):
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        return self.pi(x)
  

class SelectorLVL2(AbstractSelector):
    def __init__(self,input_size = (1,28,28), output_size = (1,28,28)):
      super().__init__(input_size = input_size, output_size = output_size)
      
      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.fc2 = nn.Linear(50,50)
      self.pi = nn.Linear(50, np.prod(self.output_size))
      


    def __call__(self, x):
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.pi(x)
  

class SelectorLVL3(AbstractSelector):
    def __init__(self,input_size = (1,28,28), output_size = (1,28,28)):
      super().__init__(input_size = input_size, output_size = output_size)

      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.fc2 = nn.Linear(50,50)
      self.fc3 = nn.Linear(50,50)
      self.fc4 = nn.Linear(50,50)
      self.pi = nn.Linear(50, np.prod(self.output_size))
      


    def __call__(self, x):
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        return self.pi(x)
  
class RealXSelector(AbstractSelector):
  def __init__(self, input_size, output_size, middle_size = 100):
    super().__init__(input_size = input_size, output_size = output_size)
    

    self.fc1 = nn.Linear(np.prod(self.input_size),middle_size)
    self.fc2 = nn.Linear(middle_size,middle_size)
    self.pi = nn.Linear(middle_size, np.prod(self.output_size))

  def __call__(self, x):
    x = x.flatten(1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.pi(x)

class Selector(AbstractSelector):
    def __init__(self, input_size = (1,28,28), output_size = (1,28,28)):
      super().__init__(input_size = input_size, output_size = output_size)
      self.fc1 = nn.Linear(np.prod(self.input_size),200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, np.prod(self.output_size))
      


    def __call__(self, x):
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.pi(x)


def calculate_blocks_patch(input_size, kernel_size, kernel_stride):
  """
  Calculates the number of blocks in a patch.
  """
  size = [1,]
  for k in range(len(kernel_size)):
    # size.append(math.floor((input_size[k+1] - (kernel_size[k]-1)) / kernel_stride[k]) + 1)
    size.append(int(np.ceil(input_size[k+1]/kernel_stride[k])))

  return tuple(size)


class SelectorUNET(AbstractSelector):
    def __init__(self,  input_size = (1,28,28), output_size= (1, 28, 28), kernel_size = (1,1), kernel_stride = (1,1), bilinear = True):

      aux_output_size = calculate_blocks_patch(input_size, kernel_size, kernel_stride)
      assert aux_output_size == output_size, "Output size of the selector must be the same as the output size of the unet."
      super().__init__(input_size = input_size, output_size = output_size)
      self.channels = self.input_size[0]
      self.w = self.input_size[1]
      self.h = self.input_size[2]
      self.bilinear = bilinear

    
      self.nb_block = int(math.log(min(self.output_size[1], self.output_size[2]), 2)//2)
      self.getconfiguration = nn.Sequential(*[
        nn.Conv2d(self.channels, 64, kernel_size = kernel_size, stride = kernel_stride),
        nn.ReLU(inplace = False),
        nn.Conv2d(64, 64, kernel_size = 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = False),
      ])

      self.UNET = UNet(1, bilinear = self.bilinear, nb_block = self.nb_block)


    def __call__(self, x):
      x = self.getconfiguration(x)
      x = self.UNET(x)
      x = x.view(-1, np.prod(self.output_size))

      return x

class SelectorUNET1D(AbstractSelector):
    def __init__(self, input_size = (22, 19), output_size = (1,28,28), bilinear = False):
      raise NotImplementedError("This selector is not implemented yet.")
      # super().__init__(input_size = input_size, output_size = output_size)
      # self.channels = self.input_size[0]
      # self.w = self.input_size[1]
      # self.bilinear = bilinear
      
      # self.nb_block = int(math.log(self.nb_patch_x, 2)//2)
      # self.getconfiguration = nn.Sequential(*[
      #   nn.Conv1d(self.channels, 64, kernel_size = kernel_patch, stride = stride_patch),
      #   nn.ReLU(inplace = False),
      #   nn.Conv1d(64, 64, kernel_size = 3, padding=1),
      #   nn.BatchNorm1d(64),
      #   nn.ReLU(inplace = False),
      #   ])


    

    # def __call__(self, x):
    #   x = self.getconfiguration(x)
    #   x = self.UNET(x)
    #   return x










class SelectorVariational(AbstractSelector):
  def __init__(self, input_size = (1,28,28), output_size = (1,28,28)):
    super().__init__(input_size = input_size, output_size = output_size)
    self.output_size = output_size
    
    

  def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)
      self.fc1 = nn.Linear(np.prod(self.input_size)+self.output_size,200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, np.prod(self.output_size))

  def __call__(self, x, y):
    assert(self.kernel_updated)
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    y = y.float()
    x = torch.cat([x,y],1)
    x = F.elu(self.fc1(x))
    pi = F.elu(self.fc2(x))
    return self.pi(pi)







class ConvSelector(nn.Module):
    def __init__(self, input_channel, input_size = (1,28,28), output_size= 10):
        super().__init__()

        
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1, padding = 1)
        self.conv2 = nn.Conv2d(input_channel, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
        self.fc = nn.Linear(np.prod(input_size),np.prod(input_size)) #TODO : No Fully connected layer at the end.
    
    def __call__(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        return self.fc(x) #N_expectation, Batch_size, Category

class ConvSelectorVar(nn.Module):
  def __init__(self, input_channel, input_size = (1,28,28), output_size= 10):
    super().__init__()
    
    self.conv1 = nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1)
    self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
    self.conv2 = nn.Conv2d(input_channel, 1, 3, stride=1, padding=1)
    self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1,padding = 1)
    self.fc = nn.Linear(input_size**2+output_size, input_size**2)
  
  def __call__(self, x, y):
    x = self.maxpool1(self.conv1(x))
    x = self.maxpool2(self.conv2(x))
    x = torch.flatten(x,1)
    y = y.float()
    x = torch.cat([x,y],1)
    return self.fc(x) #TODO : No Fully connected layer at the end.


selection_network_list = {
  "SelectorLinear": SelectorLinear,
  "SelectorREALX" : SelectorRealX,
  "SelectorUNET" : SelectorUNET,
  "SelectorLVL1" : SelectorLVL1,
  "SelectorVariational" : SelectorVariational,
  "SelectorLVL2" : SelectorLVL2,
  "SelectorLVL3" : SelectorLVL3,
  "SelectorUNET1D" : SelectorUNET1D,
  "ConvSelector" : ConvSelector,
}

def get_selection_network(selector_name):
  if selector_name is None or selector_name == "none" :
    return None
  elif selector_name in selection_network_list.keys():
    return selection_network_list[selector_name]
  else:
    raise NotImplementedError(f"This selector {selector_name} is not implemented")