import copy

from torch.distributions import *
from functools import partial


def get_distribution(distribution, temperature):


    if distribution is RelaxedBernoulli:
        current_sampling = partial(RelaxedBernoulli,temperature)
    else :
        current_sampling = copy.deepcopy(distribution)

    return current_sampling
