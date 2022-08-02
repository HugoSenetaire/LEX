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
    def __init__(self, input_size, output):
        super(PredictorAbstract, self).__init__()
        self.input_size = input_size
        self.output = output
        if self.output > 1:
            self.activation = nn.LogSoftmax(-1)
        else :
            self.activation = lambda x: x
    def forward(self, x):
        raise NotImplementedError
    
class ClassifierLinear(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), output)
    
    def __call__(self, x):
        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = self.fc1(x)
        return self.activation(x) #N_expectation * Batch_size, Category

class ClassifierLVL1(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, output)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x) #N_expectation * Batch_size, Category

class ClassifierLVL2(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, output)
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return self.activation(x) #N_expectation * Batch_size, Category


class ClassifierLVL3(PredictorAbstract):
    def __init__(self,input_size = (1,28,28), output = 10, middle_size = 50):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), middle_size)
        self.fc2 = nn.Linear(middle_size, middle_size)
        self.fc3 = nn.Linear(middle_size, middle_size)
        self.fc4 = nn.Linear(middle_size, middle_size)
        self.fc5 = nn.Linear(middle_size, output)
        
    
    def __call__(self, x):
        x = x.flatten(1)  

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return self.activation(x)

# class RealXClassifier(PredictorAbstract):
#     def __init__(self, input_size, output, middle_size=200):
#         super().__init__(input_size=input_size, output=output)
#         self.input_size = input_size
#         self.fc1 = nn.Linear(np.prod(input_size), 200)
#         self.bn1 = nn.BatchNorm1d(200)
#         self.fc2 = nn.Linear(200, 200)
#         self.bn2 = nn.BatchNorm1d(200)
#         self.fc3 = nn.Linear(200, output)


#     def __call__(self, x):
#         x = x.flatten(1)
#         x = F.relu(self.fc1(x))
#         x = self.bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.bn2(x)
#         x = self.fc3(x)

#         return self.activation(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class RealXClassifier(PredictorAbstract):
    def __init__(self, input_size, output, middle_size=200):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output)

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
    def __init__(self, input_size = (1,28,28),output = 10, bias = True):
        super().__init__(input_size=input_size, output=output)
        self.bias = bias
        self.input_size = input_size
        self.fc1 = nn.Linear(np.prod(input_size), output, bias = bias)
        
        self.elu = nn.ELU()


    def __call__(self, x):
        x = x.flatten(1)
        return self.activation(self.elu(self.fc1(x)))


class PretrainedVGGPytorch(PredictorAbstract):
    def __init__(self, input_size = (3, 224, 224), output = 2, model_type = "vgg11", pretrained= True, retrain = False):
        super().__init__(input_size=input_size, output=output)
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
                                nn.Linear(4096, output),
                            )
        self.elu = nn.ELU()

    def __call__(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.new_classifier(x)
        x = self.activation(self.elu(x))
        return x



class VGGSimilar(PredictorAbstract):
    def __init__(self, input_size = (3,224, 224), output = 2):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        assert(len(input_size)==3)
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
                                nn.Linear(4096, output)
                            )
        self.elu = nn.ELU()

        

    def __call__(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.new_classifier(x)
        x = self.activation(self.elu(x))
        return x


class ConvClassifier(PredictorAbstract):
    def __init__(self, input_size = (1,28,28), output = 10):
        super().__init__(input_size=input_size, output=output)
        self.conv1 = nn.Conv2d(input_size[0], 10, 3, stride=1, padding=1)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.conv2 = nn.Conv2d(10, 1, 3, stride=1, padding=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.fc = nn.Linear(int(np.prod(input_size[1:])/16),output)
        self.elu = nn.ELU()
    
    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.activation(self.elu(self.fc(x))).reshape(batch_size, -1)
        return result #N_expectation, Batch_size, Category



class ConvClassifier2(PredictorAbstract):
    def __init__(self, input_size = (1,28,28), output = 10):
        super().__init__(input_size=input_size, output=output)
        self.conv1 = nn.Conv2d(input_size[0], 6, 5, stride=1, padding=0)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=1,padding = 0) # 23 23
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0) # 23 23
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=1,padding = 0) # 18 18
        self.fc = nn.Linear(18*18*16,output) # 18 18
        self.elu = nn.ELU()
    
    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.flatten(x,1)
        result = self.activation(self.elu(self.fc(x))).reshape(batch_size, -1)
        return result #N_expectation, Batch_size, Category




class ProteinCNN(PredictorAbstract):
    def __init__(self, input_size = (21,19), output = 8):
        super().__init__(input_size=input_size, output=output)
        self.input_size = input_size
        self.output = output
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
            nn.Linear(32, output),
            nn.LogSoftmax(-1),
        )
    def __call__(self, x):
        x = self.cnns(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x




classifiers_list = {
    "none": None,
    "ClassifierLinear" : ClassifierLinear,
    "ConvClassifier" : ConvClassifier,
    "ConvClassifier2" : ConvClassifier2,
    "ProteinCNN" : ProteinCNN,
    "ClassifierLvl1" : ClassifierLVL1,
    "ClassifierLvl2" : ClassifierLVL2,
    "ClassifierLvl3" : ClassifierLVL3,
    "RealXClassifier" : RealXClassifier,
}


def get_pred_network(classifier_name):
    if classifier_name == "none" or classifier_name == None:
        return None
    elif classifier_name in classifiers_list:
        return classifiers_list[classifier_name]
    else:
        raise ValueError(f"Classifier {classifier_name} not found")

