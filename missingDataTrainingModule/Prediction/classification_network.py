import sys
import os

from torch.nn.modules.activation import ELU

import math
import torch
import torchvision
import torch.nn as nn
import numpy as np


from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import models

class PredictorAbstract(nn.Module):
    def __init__(self, input_size, output_size):
        super(PredictorAbstract, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        if type(self.output_size) == int:
            self.output = output_size
            if self.output > 1:
                self.activation = nn.LogSoftmax(-1)
            else :
                self.activation = lambda x: x
        else:
            self.output = np.prod(self.output_size)
            if len(self.output_size) >1:
                raise NotImplementedError
            self.activation = lambda x : x
        
    def forward(self, x):
        raise NotImplementedError
    
class ClassifierLinear(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output_size = 10, middle_size = 50):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), self.output)
    
    def __call__(self, x):
        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = self.fc1(x)
        return self.activation(x) #N_expectation * Batch_size, Category

class ClassifierLVL1(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output_size = 10, middle_size = 50):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, self.output)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x) #N_expectation * Batch_size, Category

class ClassifierLVL2(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output_size = 10, middle_size = 50):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, self.output)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return self.activation(x) #N_expectation * Batch_size, Category


class ClassifierLVL3(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output_size = 10, middle_size = 50):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, middle_size)
        self.fc4 = nn.Linear(middle_size, middle_size)
        self.fc5 = nn.Linear(middle_size, self.output)
        
    
    def __call__(self, x):
        x = x.flatten(1)  

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return self.activation(x)

class RealXClassifier_withBatchnorm(PredictorAbstract):
    def __init__(self, input_size = (1,28,28), output_size = 10, middle_size = 50):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, self.output)


    def __call__(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)

        return self.activation(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class RealXClassifier(PredictorAbstract):
    def __init__(self, input_size, output_size, middle_size=200):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.output)

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

    def __call__(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return self.activation(x)

class StupidClassifier(PredictorAbstract):
    def __init__(self, input_size = (1,28,28),output_size = 10, bias = True):
        super().__init__(input_size=input_size, output_size=output_size)
        self.bias = bias
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), self.output, bias = bias)
        
        self.elu = nn.ELU()


    def __call__(self, x):
        x = x.flatten(1)
        return self.activation(self.elu(self.fc1(x)))


class PretrainedVGGPytorch(PredictorAbstract):
    def __init__(self, input_size = (3, 224, 224), output_size = 2, model_type = "vgg19", ):
        super().__init__(input_size=input_size, output_size=output_size)
        assert(model_type.startswith("vgg"))
        self.model = torch.hub.load('pytorch/vision:v0.9.0', model_type, pretrained=True)
        self.model.requires_grad_(False)
        self.new_classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(512*7*7, 4096),
                                nn.ReLU(True),
                                nn.Dropout(),
                                nn.Linear(4096,4096),
                                nn.ReLU(True),
                                nn.Dropout(),
                                nn.Linear(4096, self.output),
                            )
        self.new_classifier.requires_grad_(True)
        self.elu = nn.ELU()

    def __call__(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.new_classifier(x)
        x = self.activation(self.elu(x))
        return x


class ConvClassifier(PredictorAbstract):
    def __init__(self, input_size = (1,28,28), output_size = 10):
        super().__init__(input_size=input_size, output_size=output_size)
        self.conv1 = nn.Conv2d(input_size[0], 10, 3, stride=1, padding=1)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.conv2 = nn.Conv2d(10, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.fc = nn.Linear(int(np.prod(input_size[1:])/16),self.output)
        self.elu = nn.ELU()
    
    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.activation(self.elu(self.fc(x)))
        return result #N_expectation, Batch_size, Category



class ConvClassifier2(PredictorAbstract):
    def __init__(self, input_size = (1,28,28), output_size = 10):
        super().__init__(input_size=input_size, output_size=output_size)
        self.nb_block = int(math.log(min(self.input_size[1], self.input_size[2]), 2)//2)
        
        liste_conv = []
        liste_conv.extend([
            nn.Conv2d(input_size[0], 2**5, 3, stride=1, padding=1),
            nn.Conv2d(2**5, 2**5, 3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        ])
        for k in range(1, self.nb_block):
            liste_conv.extend([
                nn.Conv2d(2**(k+4), 2**(k+5), 3, stride=1, padding=1),
                nn.Conv2d(2**(k+5), 2**(k+5), 3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0),
            ]
            )
        self.conv = nn.ModuleList(liste_conv)
        last_channel = 2**(self.nb_block+4)
        last_size = int(np.prod(input_size[1:])/(2**(2*self.nb_block)))
        self.fc = nn.Linear(last_channel*last_size,128)

        self.elu = nn.ELU()

        self.fc2 = nn.Linear(128,self.output)


    
    def __call__(self, x):
        batch_size = x.shape[0]
        for k in range(len(self.conv)):
            x = self.conv[k](x)
        x = x.flatten(1)
        x = self.elu(self.fc(x))
        x = self.activation(self.fc2(x))
        return x 


            

class ResNet50(PredictorAbstract):
    def __init__(self, input_size = (3, 224, 224), output_size = 10):
        super().__init__(input_size=input_size, output_size=output_size)

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(512, self.output)

    def __call__(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x

class ResNet34(PredictorAbstract):
    def __init__(self, input_size = (3, 224, 224), output_size = 10):
        super().__init__(input_size=input_size, output_size=output_size)

        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, self.output)

    def __call__(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x

class ProteinCNN(PredictorAbstract):
    def __init__(self, input_size = (21,19), output_size = 8):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.dropout_rate = 0.38 # In the original implementation https://github.com/LucaAngioloni/ProteinSecondaryStructure-CNN/

        self.cnns = nn.Sequential(
            nn.Conv1d(self.input_size[0], 128, 5, stride =1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=self.dropout_rate),

            nn.Conv1d(128,128, 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=self.dropout_rate),

            nn.Conv1d(128, 64, 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=self.dropout_rate),

        )
        self.fc = nn.Sequential(
            nn.Linear(64*self.input_size[1], 128),
            nn.ReLU(), 
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.output),
            nn.LogSoftmax(-1),
        )
    def __call__(self, x):
        x = self.cnns(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

