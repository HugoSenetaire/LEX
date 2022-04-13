
import torch
import numpy as np
def collapse_in_batch(z, size):
    """ This reshape function goal is simply to get a shape of mask where we collapsed MC IWAE and MC SAMPLE in the batch dimension. """
    new_size = torch.Size((-1,))+torch.Size(size)
    z = z.reshape(new_size)
    return z

def reshape_kernel_2D(z, input_size_classifier, output_size_selector, kernel_size, kernel_stride):
    """ Using block extracted from the network, we reshape it into a 2D Image """
    if kernel_size == (1,1) :
        return collapse_in_batch(z, input_size_classifier)
    else :
        z = z.reshape(-1, output_size_selector[0], output_size_selector[1]*output_size_selector[2]) # Collapse the mc dim in batch, flatten the number of blocks in one dim
        z = z.unsqueeze(2).expand(-1, -1, np.prod(kernel_size), -1).flatten(1,2) # 
        new_z = torch.nn.Fold((input_size_classifier[1], input_size_classifier[2]), kernel_size, kernel_stride)(z)
        new_z = new_z.clamp(0.0, 1.0)

        for k in range(2, len(input_size_classifier)):
            new_z = new_z[:, :, :input_size_classifier[k],]
            new_z.flatten(1,2)
        new_z = new_z.reshape((z.shape[0], *input_size_classifier))

        # new_z = new_z[:,:, :input_size_classifier[1], :input_size_classifier[2]]

        return new_z
