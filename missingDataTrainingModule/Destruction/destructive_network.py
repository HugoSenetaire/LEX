import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# def calculate_patch_parameters(input_size, nb_patch_total):
#     if len(input_size)==1:
#         input_size = (input_size, input_size)

#     stride_x = nb_patch_total

# Define the generator

class AbstractDestructor(nn.Module):
  def __init__(self,input_size = (1,28,28)):
    super().__init__()

    self.input_size = input_size
    
    # print(self.nb_patch_x)
    # print(self.nb_patch_y)
    
    self.kernel_updated = False
  def kernel_update(self, kernel_patch, stride_patch):
    self.kernel_updated = True
    assert(kernel_patch[0]>= stride_patch[0])
    assert(kernel_patch[1]>= stride_patch[1])
    assert(stride_patch[0]>0)
    assert(stride_patch[1]>0)
    self.kernel_patch = kernel_patch
    self.stride_patch = stride_patch
    self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(self.input_size, self.stride_patch)



  
  def __call__(self, x):
    raise NotImplementedError

class Destructor(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)
      self.fc1 = nn.Linear(np.prod(self.input_size),200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, self.nb_patch_x*self.nb_patch_y)
        


    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        pi = F.elu(self.fc2(x))
        return F.sigmoid(self.pi(pi))


class PatchDestructorV1(nn.Module):
    def __init__(self, input_size = (1,28,28), stride = (1,1), kernel_patch = (1,1)):
      super().__init__()
      self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(input_size, stride)
      self.input_size = input_size
    
    def __call__(self, x):
      raise NotImplementedError
      
      

class DestructorVariational(AbstractDestructor):
  def __init__(self, input_size = (1,28,28), output_size = 10):
    super().__init__(input_size = input_size)
    self.output_size = output_size
    

  def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)
      self.fc1 = nn.Linear(np.prod(self.input_size)+self.output_size,200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, self.nb_patch_x*self.nb_patch_y)

  def __call__(self, x, y):
    assert(self.kernel_updated)
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    y = y.float()
    x = torch.cat([x,y],1)
    x = F.elu(self.fc1(x))
    pi = F.elu(self.fc2(x))
    return F.sigmoid(self.pi(pi))


class DestructorVariationalNoY(AbstractDestructor):
  def __init__(self, input_size = (1,28,28), output_size = 10):
    super().__init__(input_size = input_size)
    self.output_size = output_size
    

  def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)
      self.fc1 = nn.Linear(np.prod(self.input_size),200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, self.nb_patch_x*self.nb_patch_y)

  def __call__(self, x, y):
    assert(self.kernel_updated)
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    y = y.float()
    # x = torch.cat([x,y],1)
    x = F.elu(self.fc1(x))
    pi = F.elu(self.fc2(x))
    return F.sigmoid(self.pi(pi))

class ConvDestructor(nn.Module):
    def __init__(self, input_channel, input_size = (1,28,28), output_size= 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1, padding = 1)
        self.conv2 = nn.Conv2d(input_channel, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
        self.fc = nn.Linear(np.prod(input_size),np.prod(input_size))
    
    def __call__(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        return F.sigmoid(self.fc(x)) #N_expectation, Batch_size, Category

class ConvDestructorVar(nn.Module):
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
    return F.sigmoid(self.fc(x)) #N_expectation, Batch_size, Category
