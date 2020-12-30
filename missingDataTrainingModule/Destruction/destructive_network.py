import torch.nn as nn
import torch.nn.functional as F
import torch

# Define the generator
class Destructor(nn.Module):
    def __init__(self,input_size = 28):
        super().__init__()
        self.fc1 = nn.Linear(input_size**2,200)
        self.pi = nn.Linear(200, input_size**2)

    def __call__(self, x):
        # print(x.shape)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        pi = F.elu(self.fc1(x))
        return F.sigmoid(self.pi(pi))

class DestructorVariational(nn.Module):
  def __init__(self, input_size = 28, output_size = 10):
    super().__init__()
    self.fc1 = nn.Linear(input_size**2+output_size,200)
    self.pi = nn.Linear(200, input_size**2)

  def __call__(self, x, y):
    # X et Y doivent avoir N_expectation ?
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    # y = torch.double(y)
    y = y.float()
 # Y : Batch_size, Ncategory
    x = torch.cat([x,y],1)
    pi = F.elu(self.fc1(x))
    return F.sigmoid(self.pi(pi))


class ConvDestructor(nn.Module):
    def __init__(self, input_channel, input_size = 28, output_size= 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1, padding = 1)
        self.conv2 = nn.Conv2d(input_channel, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
        self.fc = nn.Linear(input_size**2,input_size**2)
    
    def __call__(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        return F.sigmoid(self.fc(x)) #N_expectation, Batch_size, Category

class ConvDestructorVar(nn.Module):
  def __init__(self, input_channel, input_size = 28, output_size= 10):
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
