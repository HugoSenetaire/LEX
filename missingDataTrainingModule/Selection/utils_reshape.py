
import torch
import numpy as np


class CollapseInBatch():
    def __init__(self, size):
        self.size = size

    def __call__(self, z):
        new_size = torch.Size((-1,))+torch.Size(self.size)
        z = z.reshape(new_size)
        return z

class KernelReshape2D():
    def __init__(self, input_size_classifier, output_size_selector, kernel_size, kernel_stride):
        self.input_size_classifier = input_size_classifier
        self.output_size_selector = output_size_selector
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        if np.all(self.kernel_size == 1) :
            self.collapse = CollapseInBatch(self.input_size_classifier)


    def __call__(self, z):
        if np.all(self.kernel_size == 1) :
            return self.collapse(z)
        else :
            z = z.reshape(-1, self.output_size_selector[0], self.output_size_selector[1]*self.output_size_selector[2]) # Collapse the mc dim in batch, flatten the number of blocks in one dim
            z = z.unsqueeze(2).expand(-1, -1, np.prod(self.kernel_size), -1).flatten(1,2) # 
            new_z = torch.nn.Fold((self.input_size_classifier[1], self.input_size_classifier[2]), self.kernel_size, self.kernel_stride)(z)
            new_z = new_z.clamp(0.0, 1.0)

            for k in range(2, len(self.input_size_classifier)):
                new_z = new_z[:, :, :self.input_size_classifier[k],]
                new_z.flatten(1,2)
            new_z = new_z.reshape((z.shape[0], *self.input_size_classifier))
            # new_z = new_z[:,:, :self.input_size_classifier[1], :self.input_size_classifier[2]]

            return new_z


