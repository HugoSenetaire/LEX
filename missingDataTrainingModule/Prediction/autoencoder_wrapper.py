import torch.nn as nn
from ..Selection.selective_network import *

class AutoEncoderWrapper(nn.Module):
    def __init__(self, input_size = (1,28,28), output_size= (1, 28, 28), kernel_size = (1,1), kernel_stride = (1,1), bilinear = True, log2_min_channel = 6):
        super(AutoEncoderWrapper, self).__init__()
        self.autoencoder = SelectorUNET(input_size = input_size, output_size=output_size, bilinear = bilinear, kernel_size= kernel_size, kernel_stride = kernel_stride, log2_min_channel = log2_min_channel)
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.bilinear = bilinear
        self.log2_min_channel = log2_min_channel
        
    def __call__(self, x):
        return self.autoencoder(x)
