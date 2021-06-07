from torch import nn
from torch.optim import Adam

try :
  from mask_generators import ImageMaskGenerator, DropoutMaskGenerator
  from nn_utils import ResBlock, MemoryLayer, SkipConnection
  from prob_utils import normal_parse_params, GaussianLoss
except :
  from missingDataTrainingModule import *


# sampler from the model generative distribution
# here we return mean of the Gaussian to avoid white noise
def sampler(params, multiple=False):
    return normal_parse_params(params, multiple=multiple).mean


def optimizer(parameters):
    return Adam(parameters, lr=2e-4)


batch_size = 16

reconstruction_log_prob = GaussianLoss()

mask_generator = ImageMaskGenerator(input_size=28)

# improve train computational stability by dividing the loss
# by this scale factor right before backpropagation
vlb_scale_factor = 28 ** 2
class StupidLayer(nn.Module):

    def __init__(self):
        super(StupidLayer, self).__init__()

    def forward(self,x):
        return x[:,:,2:-2,2:-2]

def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm2d(dim),
        nn.LeakyReLU(),
        nn.Conv2d(dim, dim, 1)
    )

proposal_network = nn.Sequential(
    nn.Conv2d(2, 8, 1,padding=2), #28,28,8
    ResBlock(8, 8), ResBlock(8, 8), 
    nn.AvgPool2d(2, 2), # 16, 16,8
    ResBlock(8, 8), ResBlock(8, 8), 
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1), # 8, 8, 16
    ResBlock(16, 8), ResBlock(16, 8), # 8,8, 16?
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 4, 4, 32
    ResBlock(32, 16), ResBlock(32, 16),

    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), # 2,2 64
    ResBlock(64, 32), ResBlock(64, 32),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
    MLPBlock(128), MLPBlock(128), 
)

prior_network = nn.Sequential(
    MemoryLayer('#0'),
    nn.Conv2d(2, 8, 1, padding=2), # 28,28,8
    ResBlock(8, 8), ResBlock(8, 8), 
    MemoryLayer('#1'),
    nn.AvgPool2d(2, 2),# 16,16,8
    ResBlock(8, 8), ResBlock(8, 8), 
    MemoryLayer('#2'),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),# 8,8,16
    ResBlock(16, 8), ResBlock(16, 8),
    MemoryLayer('#3'),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1), # 4,4 ,32
    ResBlock(32, 16), ResBlock(32, 16), 
    MemoryLayer('#4'),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1), #2,2 64
    ResBlock(64, 32), ResBlock(64, 32),
    MemoryLayer('#5'),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1), #1,1,128
    MLPBlock(128), MLPBlock(128),
)

generative_network = nn.Sequential(
    nn.Conv2d(64, 64, 1),
    MLPBlock(64), MLPBlock(64),
    nn.Conv2d(64, 32, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#5', True), nn.Conv2d(96, 32, 1),
    ResBlock(32, 16), ResBlock(32, 16), 
    nn.Conv2d(32, 16, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#4', True), nn.Conv2d(48, 16, 1),
    ResBlock(16, 8), ResBlock(16, 8), 
    nn.Conv2d(16, 8, 1), nn.Upsample(scale_factor=2),
    MemoryLayer('#3', True), nn.Conv2d(24, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), 
    nn.Upsample(scale_factor=2),
    MemoryLayer('#2', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), 
    nn.Upsample(scale_factor=2), #32,32,8

    # nn.Conv2dTranspose(8,8,stride=2,padding=1) 
    MemoryLayer('#1', True), nn.Conv2d(16, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), 
    StupidLayer(),
    MemoryLayer('#0', True), nn.Conv2d(10, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), 
    nn.Conv2d(8, 2, 1),

)
