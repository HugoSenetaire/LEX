
import torch

def simple_reshape(z, size):
    new_size = torch.Size((-1,))+torch.Size(size)
    z = z.reshape(new_size)
    return z

    