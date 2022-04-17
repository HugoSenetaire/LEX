import torch
from torch.utils.data import TensorDataset
import numpy as np 


class TensorDatasetAugmented(TensorDataset):
    def __init__(self, x, y, noise_function = None, give_index = False):
        super().__init__(x,y)
        self.noise_function = noise_function
        self.give_index = give_index 


    def __getitem__(self, idx):

        input_tensor, target = self.tensors[0][idx].type(torch.float32), self.tensors[1][idx].type(torch.int64)    
        input_tensor = input_tensor.numpy()
        target = target.numpy()
        if self.noise_function is not None :
            input_tensor = torch.tensor(self.noise_function(input_tensor)).type(torch.float32)

        if self.give_index :
            return input_tensor, target, idx
        else :
            return input_tensor, target


