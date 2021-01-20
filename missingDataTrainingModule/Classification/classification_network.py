import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *


import torch
import torchvision
import torch.nn as nn


from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



class ClassifierModel(nn.Module):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 400):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size,output)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):
        
        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsoftmax(self.fc3(x)) #N_expectation * Batch_size, Category



class ConvClassifier(nn.Module):
    def __init__(self, output_size = 10, input_size = (1,28,28)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], input_size[0], 3, stride=1, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.conv2 = nn.Conv2d(input_size[0], 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2,padding = 0)
  
        self.fc = nn.Linear(int(input_size/4)**2,output_size)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):
        batch_size = x.shape[0]
        # batch_size = x.shape[1]
        # x = torch.flatten(x,0,1)
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.logsoftmax(self.fc(x)).reshape(batch_size, -1)
        return result #N_expectation, Batch_size, Category


class imputationInvariantNetwork(nn.Module):
    def __init__(self, input_size = (50), output_size = 20):
        super().__init__()
        self.fc =  nn.Linear(np.prod(input_size), output_size)

    def __call__(self, x):
        return F.elu(self.fc(x))
    

class AutoEncoder(nn.Module):
    def __init__(self, input_size = (1, 28, 28), output_size = 20):
        super().__init__()
        input_size_flatten = np.prod(input_size)

        self.fc1 = nn.Linear(input_size_flatten, int(input_size_flatten/2))
        self.fc2 = nn.Linear(int(input_size_flatten/2), int(input_size_flatten/4))
    
        self.fc2_output = nn.Linear(int(input_size_flatten/4), int(input_size_flatten/2))
        self.fc1_output = nn.Linear(int(input_size_flatten/2), input_size_flatten)


    def __call__(self, x):
        x = x.flatten(1)
        # print(x.shape)
        # print(x)
        # print(self.fc1.weight)
        # print(self.fc1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc2_output(x))
        x = F.elu(self.fc1_output(x))

        return x