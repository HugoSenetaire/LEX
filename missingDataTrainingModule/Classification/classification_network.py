import sys
import os

from torch.nn.modules.activation import ELU
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torchvision
import torch.nn as nn
import numpy as np


from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image




    
class ClassifierLinear(nn.Module):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), output)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):
        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = self.fc1(x)
        return self.logsoftmax(x) #N_expectation * Batch_size, Category

class ClassifierLVL1(nn.Module):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, output)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return self.logsoftmax(x) #N_expectation * Batch_size, Category

class ClassifierLVL2(nn.Module):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, output)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return self.logsoftmax(x) #N_expectation * Batch_size, Category


class ClassifierLVL3(nn.Module):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, middle_size)
        self.fc4 = nn.Linear(middle_size, middle_size)
        self.fc5 = nn.Linear(middle_size, output)
        
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):
        x = x.flatten(1)  

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return self.logsoftmax(x)

class RealXClassifier(nn.Module):
    def __init__(self, input_size, output, middle_size=200):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.bn1 = nn.BatchNorm1d(middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.bn2 = nn.BatchNorm1d(middle_size)
        self.fc3 = nn.Linear(middle_size, output)

        self.logsoftmax = nn.LogSoftmax(-1)

    def __call__(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)

        return self.logsoftmax(x)

class StupidClassifier(nn.Module):
    def __init__(self, input_size = (1,28,28),output = 10, bias = True):
        super().__init__()
        self.bias = bias
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), output, bias = bias)
        
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()


    def __call__(self, x):
        x = x.flatten(1)
        return self.logsoftmax(self.elu(self.fc1(x)))


class PretrainedVGGPytorch(nn.Module):
    def __init__(self, input_size = (3, 224, 224), output_size = 2, model_type = "vgg11", pretrained= True, retrain = False):
        super().__init__()
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
                                nn.Linear(4096, output_size),
                            )
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()

    def __call__(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.new_classifier(x)
        x = self.logsoftmax(self.elu(x))
        return x



class VGGSimilar(nn.Module):
    def __init__(self, input_size = (3,224, 224), output_size = 2):
        super().__init__()
        self.input_size = input_size
        assert(len(input_size)==3)
        self.output_size = output_size
        w = input_size[1]
        self.nb_block = min(int(math.log(w/7., 2)),5)
        if self.nb_block<1 :
            raise EnvironmentError("Size of the image should be higher than 7")


        list_feature = []
        in_channels = input_size[0]
        for k in range(self.nb_block):
            out_channels = 2**(5+k)
            list_feature.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            list_feature.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            list_feature.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            list_feature.append(nn.ReLU(inplace=True))
            list_feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*list_feature)

        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        self.new_classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(out_channels*7*7, 4096),
                                nn.ReLU(True),
                                nn.Dropout(),
                                nn.Linear(4096,4096),
                                nn.ReLU(True),
                                nn.Dropout(),
                                nn.Linear(4096, output_size)
                            )
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()

        

    def __call__(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.new_classifier(x)
        x = self.logsoftmax(self.elu(x))
        return x


class ConvClassifier(nn.Module):
    def __init__(self, input_size = (1,28,28),output_size = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], 10, 3, stride=1, padding=1)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.conv2 = nn.Conv2d(10, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.fc = nn.Linear(int(np.prod(input_size)/16),output_size)
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()
    
    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.logsoftmax(self.elu(self.fc(x))).reshape(batch_size, -1)
        return result #N_expectation, Batch_size, Category




class ProteinCNN(nn.Module):
    def __init__(self, input_size = (21,19), output_size = 8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
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
            nn.Linear(32, output_size),
            nn.LogSoftmax(-1),
        )
    def __call__(self, x):
        x = self.cnns(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
