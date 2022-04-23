
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
    def __init__(self, input_size_classifier, output_size_selector, kernel_size, kernel_stride) :
        self.input_size_classifier = input_size_classifier
        self.output_size_selector = output_size_selector
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.just_collapse = False
        if kernel_size is None or kernel_stride is None:
            self.just_collapse = True
            if self.output_size_selector[0] == self.input_size_classifier[0] : # Checking for the number of channel to be the same.
                self.collapse = CollapseInBatch(self.input_size_classifier)
            else :
                self.current_collapse = CollapseInBatch(self.output_size_selector)
                self.collapse = lambda z: CollapseInBatch(z).expand(-1, *self.input_size_classifier)
        else :
            self.fold = torch.nn.Fold( output_size=(self.input_size_classifier[1], self.input_size_classifier[2]), kernel_size = self.kernel_size, stride = self.kernel_stride)
            # if np.all(self.kernel_size == 1) :
                # self.collapse = CollapseInBatch(self.input_size_classifier)


    def __call__(self, z) :
        if self.just_collapse :
            return self.collapse(z)
        else :
            z = z.reshape(-1, self.output_size_selector[0], self.output_size_selector[1]*self.output_size_selector[2]) # Collapse the mc dim in batch, flatten the number of blocks in one dim
            z = z.unsqueeze(2).expand(-1, -1, np.prod(self.kernel_size), -1).flatten(1,2) # 
            new_z = self.fold(z)
            new_z = new_z.clamp(0.0, 1.0)
            # Following lines controls when the reconstruction is out of bounds
            for k in range(2, len(self.input_size_classifier)):
                new_z = new_z[:, :, :self.input_size_classifier[k],]
                new_z.flatten(1,2)

            if self.output_size_selector[0] == self.input_size_classifier[0] :
                new_z = new_z.reshape(-1, *self.input_size_classifier)
            else :
                if self.output_size_selector[0] != 1 :
                    raise ValueError("The output size of the selector is not 1 nor the same as the input size of the classifier, it does not make sense.")
                new_z = new_z.reshape((z.shape[0], 1, *self.input_size_classifier[1:]))
                new_z = new_z.expand((z.shape[0], *self.input_size_classifier))

            return new_z


