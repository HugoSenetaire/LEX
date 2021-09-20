import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *
from .utils_UNET import *

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np




class AbstractDestructor(nn.Module):
  def __init__(self,input_size = (1,28,28)):
    super().__init__()

    self.input_size = input_size
    self.kernel_updated = False


  def kernel_update(self, kernel_patch, stride_patch):
    self.kernel_updated = True
    
    if self.input_size is int or len(self.input_size)<=1:
      self.kernel_patch = 1
      self.stride_patch = 1
      try :
        self.nb_patch_x, self.nb_patch_y = int(self.input_size), 1
      except :
        self.nb_patch_x, self.nb_patch_y = int(self.input_size[1]), 1
    elif len(self.input_size)==2: # For protein like example (1D CNN) #TODO: really implement that ?
        self.kernel_patch = kernel_patch
        self.stride_patch = stride_patch
        self.nb_patch_x, self.nb_patch_y = int(self.input_size[1]), 1 

    else :
      assert(kernel_patch[0]>= stride_patch[0])
      assert(kernel_patch[1]>= stride_patch[1])
      assert(stride_patch[0]>0)
      assert(stride_patch[1]>0)
      self.kernel_patch = kernel_patch
      self.stride_patch = stride_patch
      self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(self.input_size, self.stride_patch)
    
  
  def __call__(self, x):
    raise NotImplementedError


class DestructorLinear(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.pi = nn.Linear(np.prod(self.input_size), self.nb_patch_x*self.nb_patch_y)
      self.logsigmoid = nn.LogSigmoid()

    def __call__(self, x):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        return self.logsigmoid(self.pi(x))

class DestructorLVL1(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.pi = nn.Linear(50, self.nb_patch_x*self.nb_patch_y)
      self.logsigmoid = nn.LogSigmoid()


    def __call__(self, x):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        return self.logsigmoid(self.pi(x))
  

class DestructorLVL2(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.fc2 = nn.Linear(50,50)
      self.pi = nn.Linear(50, self.nb_patch_x*self.nb_patch_y)
      self.logsigmoid = nn.LogSigmoid()


    def __call__(self, x):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))
  

class DestructorLVL3(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),50)
      self.fc2 = nn.Linear(50,50)
      self.fc3 = nn.Linear(50,50)
      self.fc4 = nn.Linear(50,50)
      self.pi = nn.Linear(50, self.nb_patch_x*self.nb_patch_y)
      self.logsigmoid = nn.LogSigmoid()


    def __call__(self, x):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        return self.logsigmoid(self.pi(x))
  

class Destructor(AbstractDestructor):
    def __init__(self,input_size = (1,28,28)):
      super().__init__(input_size = input_size)
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, self.nb_patch_x*self.nb_patch_y)
      self.logsigmoid = nn.LogSigmoid()


    def __call__(self, x):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))

class DestructorSimilar(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bias = True):
      super().__init__(input_size = input_size)
      self.bias = bias
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),400, bias= self.bias)
      self.fc2 = nn.Linear(400,200, bias= self.bias)
      self.fc3 = nn.Linear(200,500, bias= self.bias)
      self.pi = nn.Linear(500, self.nb_patch_x*self.nb_patch_y, bias = self.bias)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        pi = F.elu(self.fc3(x))
        return self.logsigmoid(self.pi(pi))

class DestructorSimple(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bias = True):
      super().__init__(input_size = input_size)
      self.bias = bias
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),20, bias= self.bias)
      # self.fc2 = nn.Linear(10, 10, bias= self.bias)
      self.pi = nn.Linear(20, self.nb_patch_x*self.nb_patch_y, bias = self.bias)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        # x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))

class DestructorSimpleV2(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bias = True):
      super().__init__(input_size = input_size)
      self.bias = bias
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),40, bias= self.bias)
      self.fc2 = nn.Linear(40, 40, bias= self.bias)
      self.pi = nn.Linear(40, self.nb_patch_x*self.nb_patch_y, bias = self.bias)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))




class DestructorSimpleV2(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bias = True):
      super().__init__(input_size = input_size)
      self.bias = bias
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),20, bias= self.bias)
      self.fc2 = nn.Linear(20, 20, bias= self.bias)
      self.pi = nn.Linear(20, self.nb_patch_x*self.nb_patch_y, bias = self.bias)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))

class DestructorSimpleV3(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bias = True):
      super().__init__(input_size = input_size)
      self.bias = bias
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size),50, bias= self.bias)
      self.fc2 = nn.Linear(50, 50, bias= self.bias)
      self.pi = nn.Linear(50, self.nb_patch_x*self.nb_patch_y, bias = self.bias)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsigmoid(self.pi(x))


class DestructorUNET(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), bilinear = True):
      super().__init__(input_size = input_size)
      self.channels = self.input_size[0]
      self.w = self.input_size[1]
      self.h = self.input_size[2]
      self.bilinear = bilinear
      self.logsigmoid = nn.LogSigmoid()


    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update(kernel_patch, stride_patch)
      self.nb_block = int(math.log(min(self.nb_patch_x, self.nb_patch_y), 2)//2)
      self.getconfiguration = nn.Sequential(*[
        nn.Conv2d(self.channels, 64, kernel_size = kernel_patch, stride = stride_patch),
        nn.ReLU(inplace = False),
        nn.Conv2d(64, 64, kernel_size = 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = False),
      ])

      self.UNET = UNet(1, bilinear = self.bilinear, nb_block = self.nb_block)

    def __call__(self, x):
      x = self.getconfiguration(x)
      x = self.UNET(x)
      pi = self.logsigmoid(x)
      return pi

class DestructorUNET1D(AbstractDestructor):
    def __init__(self, input_size = (22, 19), bilinear = False):
      super().__init__(input_size = input_size)
      self.channels = self.input_size[0]
      self.w = self.input_size[1]
      self.bilinear = bilinear
      self.logsigmoid = nn.LogSigmoid()


    def kernel_update(self, kernel_patch = 1, stride_patch = 1):
      super().kernel_update(kernel_patch, stride_patch)
      self.nb_block = int(math.log(self.nb_patch_x, 2)//2)
      self.getconfiguration = nn.Sequential(*[
        nn.Conv1d(self.channels, 64, kernel_size = kernel_patch, stride = stride_patch),
        nn.ReLU(inplace = False),
        nn.Conv1d(64, 64, kernel_size = 3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace = False),
      ])

      self.UNET = UNet1D(1, bilinear = self.bilinear, nb_block = self.nb_block)

    def __call__(self, x):
      x = self.getconfiguration(x)
      x = self.UNET(x)
      pi = self.logsigmoid(x)
      return pi



class DestructorSimilarVar(AbstractDestructor):
    def __init__(self,input_size = (1,28,28), nb_category = 10):
      super().__init__(input_size = input_size)
      self.nb_category = nb_category
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.fc1 = nn.Linear(np.prod(self.input_size) + self.nb_category,400)
      self.fc2 = nn.Linear(400,200)
      self.fc3 = nn.Linear(200,500)
      self.pi = nn.Linear(500, self.nb_patch_x*self.nb_patch_y)
        


    def __call__(self, x, y):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        y = y.flatten(1)
        x = torch.cat([x,y],1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        pi = F.elu(self.fc3(x))
        return self.logsigmoid(self.pi(pi))



class DestructorFromFeature(AbstractDestructor):
    def __init__(self,feature_size = [200, 500], input_size = (1,28,28)):
      super().__init__(input_size = input_size)
      self.feature_size = feature_size
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.layers = []
      for i in range(len(self.feature_size)-1):
        self.layers.append(nn.Linear(self.feature_size[i], self.feature_size[i+1]))
      self.pi = nn.Linear(self.feature_size[-1],self.nb_patch_x*self.nb_patch_y)
        
      self.module = torch.nn.ModuleList(self.layers)

    def __call__(self, x):
        # print(x.shape)
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        for layer in self.layers :
          x = F.elu(layer(x))
        return self.logsigmoid(self.pi(x))



class DestructorFromFeatureVar(AbstractDestructor):
    def __init__(self,feature_size = [200, 500], input_size = (1,28,28), nb_category = 10):
      super().__init__(input_size = input_size)
      self.nb_category =nb_category
      self.feature_size = feature_size
      self.logsigmoid = nn.LogSigmoid()
        
    def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)

      self.layers = []
      for i in range(len(self.feature_size)-1):
        self.layers.append(nn.Linear(self.feature_size[i]+self.nb_category, self.feature_size[i+1]))
      self.pi = nn.Linear(self.feature_size[-1],self.nb_patch_x*self.nb_patch_y)
        
      self.module = torch.nn.ModuleList(self.layers)

    def __call__(self, x, y):
        assert(self.kernel_updated)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        y = y.flatten(1)
        x = torch.cat([x,y], 1)
        for layer in self.layers :
          x = F.elu(layer(x))
        return self.logsigmoid(self.pi(x))


      





class DestructorVariational(AbstractDestructor):
  def __init__(self, input_size = (1,28,28), output_size = 10):
    super().__init__(input_size = input_size)
    self.output_size = output_size
    self.logsigmoid = nn.LogSigmoid()
    

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
    return self.logsigmoid(self.pi(pi))




class DestructorVariationalNoY(AbstractDestructor):
  def __init__(self, input_size = (1,28,28), output_size = 10):
    super().__init__(input_size = input_size)
    self.output_size = output_size
    self.logsigmoid = nn.LogSigmoid()
    

  def kernel_update(self, kernel_patch, stride_patch):
      super().kernel_update( kernel_patch, stride_patch)
      self.fc1 = nn.Linear(np.prod(self.input_size),200)
      self.fc2 = nn.Linear(200,100)
      self.pi = nn.Linear(100, self.nb_patch_x*self.nb_patch_y)

  def __call__(self, x, y):
    assert(self.kernel_updated)
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    y = y.float()
    x = F.elu(self.fc1(x))
    pi = F.elu(self.fc2(x))
    return self.logsigmoid(self.pi(pi))






class ConvDestructor(nn.Module):
    def __init__(self, input_channel, input_size = (1,28,28), output_size= 10):
        super().__init__()

        self.logsigmoid = nn.LogSigmoid()
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1, padding = 1)
        self.conv2 = nn.Conv2d(input_channel, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
        self.fc = nn.Linear(np.prod(input_size),np.prod(input_size)) #TODO : No Fully connected layer at the end.
    
    def __call__(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        return self.logsigmoid(self.fc(x)) #N_expectation, Batch_size, Category

class ConvDestructorVar(nn.Module):
  def __init__(self, input_channel, input_size = (1,28,28), output_size= 10):
    super().__init__()
    self.logsigmoid = nn.LogSigmoid()
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
    return self.logsigmoid(self.fc(x)) #TODO : No Fully connected layer at the end.
