from torch import nn
from torch.optim import Adam

from missingDataTrainingModule import *


# sampler from the model generative distribution
# here we return mean of the Gaussian to avoid white noise
def sampler(params, multiple = True):
    return normal_parse_params(params, multiple = multiple).mean


def optimizer(parameters):
    return Adam(parameters, lr=2e-4)


batch_size = 16

reconstruction_log_prob = GaussianLoss()

mask_generator = DropoutMaskGenerator(rate=0.8)

# improve train computational stability by dividing the loss
# by this scale factor right before backpropagation
vlb_scale_factor = 2

def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm1d(dim),
        nn.LeakyReLU(),
        nn.Linear(dim, dim)
    )

proposal_network = nn.Sequential(
    nn.Linear(4,20),
    nn.LeakyReLU(),
    nn.Linear(20,20),
    nn.LeakyReLU(),
    nn.Linear(20,20)
    # MLPBlock(20), MLPBlock(20), MLPBlock(20),
)

prior_network = nn.Sequential(
    nn.Linear(4,20),
    nn.LeakyReLU(),
    nn.Linear(20,20),
    nn.LeakyReLU(),
    nn.Linear(20,20)
    # MLPBlock(20), MLPBlock(20), MLPBlock(20),
)

generative_network = nn.Sequential(
    nn.Linear(10,10),
    MLPBlock(10), MLPBlock(10),
    nn.Linear(10,4)
)
