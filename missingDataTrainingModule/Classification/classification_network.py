
import torch
import torchvision
import torch.nn as nn


from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



class ClassifierModel(nn.Module):
    def __init__(self,input_size, output):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400,output)
        self.logsoftmax = nn.LogSoftmax(-1)
    
    def __call__(self, x):
        x = x.flatten(2)  # N_expectation, Batch_size, Channels, SizeProduct
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.logsoftmax(self.fc3(x)) #N_expectation, Batch_size, Category

        