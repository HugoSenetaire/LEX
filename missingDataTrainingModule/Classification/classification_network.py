import sys
import os

from torch.nn.modules.activation import ELU
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *

import math
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
        self.fc2 = nn.Linear(middle_size, int(middle_size/2))
        self.fc3 = nn.Linear(int(middle_size/2),output)
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()
    
    def __call__(self, x):

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsoftmax(self.elu(self.fc3(x))) #N_expectation * Batch_size, Category

    
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

        x = x.flatten(1)  # Nexpec* Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return self.logsoftmax(x)


class FeatureExtraction(nn.Module):
    def __init__(self, input_size = (1,28,28), middle_size = [400,200]):
        super().__init__()
        self.middle_size = middle_size
        self.input_size = input_size
        self.layers = []
        previous_size = np.prod(self.input_size)
        for layer in middle_size :
            self.layers.append(nn.Linear(previous_size, layer))
            previous_size = layer
        self.modules = torch.nn.ModuleList(self.layers)
    

    def __call__(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = F.elu(layer(x))
        return x


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

class ClassifierFromFeature(nn.Module):
    
    def __init__(self, input_size = 200, output_size = 10 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()
    def __call__(self, x):
        return self.logsoftmax(self.elu(self.fc(x)))

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


class ConvClassifierV2(nn.Module):
    def __init__(self, input_size = (1,28,28),output_size = 10, nb_filters = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], nb_filters, 3, stride=1, padding=1)
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding = 0)
        self.fc1 = nn.Linear(int(np.prod(input_size)/4)*nb_filters, 100)
        self.fc2 = nn.Linear(100,output_size)
        self.logsoftmax = nn.LogSoftmax(-1)
        self.elu = nn.ELU()
    
    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.maxpool1(self.conv1(x))
        x = torch.flatten(x,1)
        x=F.elu(self.fc1(x), inplace = False)
        result = self.logsoftmax(self.elu(self.fc2(x))).reshape(batch_size, -1)
        return result #N_expectation, Batch_size, Category


class AutoEncoder(nn.Module):
    def __init__(self, input_size = (1, 28, 28), output_size = 20):
        super().__init__()
        input_size_flatten = np.prod(input_size)

        self.fc1 = nn.Linear(input_size_flatten, int(input_size_flatten/2))
        self.fc2 = nn.Linear(int(input_size_flatten/2), int(input_size_flatten/4))
    
        self.fc2_output = nn.Linear(int(input_size_flatten/4), int(input_size_flatten/2))
        self.fc1_output = nn.Linear(int(input_size_flatten/2), input_size_flatten)


    def __call__(self, x):
        input_shape = x.shape
        x = x.flatten(1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc2_output(x))
        x = F.elu(self.fc1_output(x))
        x = x.reshape(input_shape)
        return x

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
