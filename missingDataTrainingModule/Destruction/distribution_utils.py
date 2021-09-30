import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
from .subsetSampling import *

import torch

### TOP K Distribution The following is not correct
class topK_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        # ctx.save_for_backward(input, k)
        _, subset_size_indices = input.topk(k, dim=-1, largest=True, sorted=False)
        output = torch.zeros(input.shape, dtype=input.dtype).scatter_(-1, subset_size_indices, 1.0)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, k = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input[input.topk(k, dim=1, largest=True, sorted=True)[1] == 0] = 0
        return grad_output, None


class RelaxedSubsetSampling_STE(RelaxedSubsetSampling):
    def __init__(self, temperature, probs=None, logits=None, subset_size = None, validate_args=None):
        super(RelaxedSubsetSampling_STE, self).__init__(temperature, probs, logits, subset_size, validate_args)

    def rsample(self, sample_shape=torch.Size()):
        sample = super().rsample(sample_shape=sample_shape)
        return topK_STE.apply(sample, self.subset_size)

### L2X Distribution and Relaxed


class L2X_Distribution(torch.distributions.RelaxedOneHotCategorical):
  def __init__(self, temperature=1, probs = None, logits = None, subset_size=None, validate_args=None):
        super(L2X_Distribution, self).__init__(temperature, probs, logits, validate_args)
        self.subset_size = subset_size


  def rsample(self, n_samples):
        samples = super(L2X_Distribution, self).rsample(n_samples).unsqueeze(0)
        for k in range(self.subset_size-1):
            samples = torch.cat((samples, super(L2X_Distribution, self).rsample(n_samples).unsqueeze(0)), 0)
        samples = torch.max(samples, dim=0)
        return samples      



    


class argmax_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input, k)
        index = torch.argmax(input, dim=-1, keepdim=True)
        
        aux = torch.zeros_like(input).scatter_(-1, index, torch.ones(input.shape, dtype=input.dtype))
        return torch.clamp(torch.sum(aux, dim=0), min=0, max=1) # Clamp is needed to get one-hot vector

    @staticmethod
    def backward(ctx, grad_output):
        # input, k = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input[input.topk(k, dim=1, largest=True, sorted=True)[1] == 0] = 0
        return grad_output, None




class L2X_Distribution_STE(torch.distributions.RelaxedOneHotCategorical):
    def __init__(self, temperature=1, probs = None, logits = None, subset_size=None, validate_args=None):
        super(L2X_Distribution_STE, self).__init__(temperature, probs, logits, validate_args)
        self.subset_size = subset_size

    def rsample(self, n_samples):
        samples = super().rsample(n_samples).unsqueeze(0)
        for k in range(self.subset_size-1):
            samples = torch.cat((samples, super().rsample(n_samples).unsqueeze(0)), 0)
        samples = argmax_STE.apply(samples)
        return samples      

    def log_prob(self, value):
        raise NotImplementedError()

    def prob(self, value):
        raise NotImplementedError()


        
### Threshold Relaxed Bernoulli


class threshold_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio):
        # ctx.save_for_backward(input, k)
        return torch.where(input > ratio, torch.ones(input.shape, dtype=input.dtype), torch.zeros(input.shape, dtype=input.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        # input, k = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input[input.topk(k, dim=1, largest=True, sorted=True)[1] == 0] = 0
        return grad_output, None


class RelaxedBernoulli_thresholded_STE(torch.distributions.RelaxedBernoulli):
    def __init__(self, temperature = 1, probs =None, logits = None, threshold = 0.5, validate_args = None) -> None:
        super(RelaxedBernoulli_thresholded_STE, self).__init__(temperature, probs, logits, validate_args)
        self.threshold = threshold

    def rsample(self, n_sample):
        samples = super(RelaxedBernoulli_thresholded_STE, self).rsample(n_sample)
        samples = threshold_STE.apply(samples, self.threshold)
        return samples


#### UTILS 


import copy

from torch.distributions import *
from functools import partial


def get_distribution(distribution, temperature, args_train):

    if distribution in [RelaxedBernoulli, RelaxedBernoulli_thresholded_STE, RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2X_Distribution_STE, L2X_Distribution]:
        current_sampling = partial(distribution, temperature = temperature)
    else :
        current_sampling = distribution

    if distribution in [SubsetSampling, RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2X_Distribution_STE, L2X_Distribution]:
        current_sampling = partial(current_sampling, subset_size = args_train["sampling_subset_size"])
    
    if distribution in [RelaxedBernoulli_thresholded_STE]:
        current_sampling = partial(current_sampling, threshold = args_train["sampling_threshold"])



    return current_sampling
