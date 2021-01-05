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
    def __init__(self,input_size = (1,28,28), output = 10):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400,output)
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
        Nexpectation = x.shape[0]
        batch_size = x.shape[1]
        # x = torch.flatten(x,0,1)
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.logsoftmax(self.fc(x)).reshape(Nexpectation, batch_size, -1)
        return result #N_expectation, Batch_size, Category