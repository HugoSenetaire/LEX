import sys
import os
from torch.distributions import distribution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from utils_missing import *
from .distribution_utils import *
from .REBAR_utils import *
from .subsetSampling import *


##### Scheduler parameters example :



def regular_scheduler(temperature, epoch, cste = 0.5):
    return temperature * cste


def get_distribution_module_from_args(args_distribution_module):

    assert(args_distribution_module["distribution"] is not None)
    if args_distribution_module["distribution_module"] is REBAR_Distribution :
        assert(args_distribution_module["distribution_relaxed"] is not None)
    # print( args_distribution_module["distribution_module"])

    distribution_module = args_distribution_module["distribution_module"](**args_distribution_module)
    # distribution_module = args_distribution_module["distribution_module"](distribution = args_distribution_module["distribution"], distribution_relaxed = args_distribution_module["distribution_relaxed"],)
    # print("INSIDE")
    # print(distribution_module)
    # print(distribution_module.distribution)
    return distribution_module


#### Distribution Module



class DistributionModule(nn.Module):
    def __init__(self, distribution, antitheis_sampling = False, **kwargs):
        super().__init__()
        self.antitheis_sampling = antitheis_sampling
        print(antitheis_sampling)
        self.distribution = distribution
        self.current_distribution = distribution

    def forward(self, log_distribution_parameters,):

        self.current_distribution = self.distribution(torch.exp(log_distribution_parameters))
        return self.current_distribution

    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)


    def sample(self, sample_shape= (1,)):
        if self.antitheis_sampling :
            # aux_sample_shape = sample_shape

            if sample_shape[-1] == 1 and self.training :
                raise(AttributeError("Antitheis sampling only works for nb_sample_z > 1"))
            
            aux_sample_shape = torch.Size(sample_shape[:-1]) + torch.Size((sample_shape[-1] // 2,) )

            # print(aux_sample_shape)
            # print(sample_shape)
            sample = self.sample_function(aux_sample_shape)
            sample = torch.cat((sample, torch.ones_like(sample)-sample), dim = len(sample_shape)-1)

            if sample_shape[-1] %2 == 1:
                rest_sample_shape = torch.Size(sample_shape[:-1]) + torch.Size((1,) )
                sample_rest = self.sample_function(rest_sample_shape)
                sample = torch.cat((sample, sample_rest), dim = len(sample_shape) - 1)
            return sample
        else :
            return self.sample_function(sample_shape)



    def update_distribution(self, epoch = None):
        return None


class DistributionWithSchedulerParameter(DistributionModule):
    def __init__(self, distribution, temperature_init = 1.0, scheduler_parameter = regular_scheduler, test_temperature = 0.0, antitheis_sampling=False,  **kwargs):
        super(DistributionWithSchedulerParameter, self).__init__(distribution, antitheis_sampling= antitheis_sampling)
        self.current_distribution = None
        self.temperature_total = temperature_init
        self.temperature = None
        self.test_temperature = test_temperature
        self.scheduler_parameter = scheduler_parameter


    def forward(self, distribution_parameters):
        self.current_distribution = self.distribution(torch.exp(distribution_parameters), temperature = self.temperature)
        return self.current_distribution

    def eval(self,):
        self.temperature = self.test_temperature
    
    def train(self,):
        self.temperature = self.temperature_total 

    def sample_function(self, sample_shape):
        if self.training :
            return self.current_distribution.rsample(sample_shape)
        else :
            return self.current_distribution.sample(sample_shape)

    def update_distribution(self, epoch = None):
        if self.scheduler_parameter is not None :
            self.temperature_total = self.scheduler_parameter(self.temperature_total, epoch)





class REBAR_Distribution(DistributionModule):
    def __init__(self, distribution, distribution_relaxed, temperature_init = 1.0, trainable = False, antitheis_sampling = False, **kwargs):
        super(REBAR_Distribution, self).__init__(distribution, antitheis_sampling=antitheis_sampling)
        if self.antitheis_sampling :
            raise AttributeError("Antitheis sampling only works for regular distribution")
        self.distribution_relaxed = distribution_relaxed
        self.trainable = trainable
        self.temperature_total = torch.nn.Parameter(torch.tensor(temperature_init), requires_grad = trainable)

    def forward(self, distribution_parameters):
        self.current_distribution = self.distribution(torch.exp(distribution_parameters),)
        self.current_distribution_relaxed = self.distribution_relaxed(torch.exp(distribution_parameters), )
        self.distribution_parameters = distribution_parameters
        return self.current_distribution, self.current_distribution_relaxed

    def sample(self, sample_shape= (1,)):

        nb_sample = np.prod(sample_shape)
        shape_distribution_parameters = self.distribution_parameters.shape()

        pi_list_extended = torch.cat([self.distribution_relaxed for k in range(nb_sample)], axis=0).reshape(torch.Size(sample_shape,)+ torch.Size(shape_distribution_parameters))
        # TODO
        # Need to extend pi_list a proper way. How ? TO TEST
        
        # u = torch.FloatTensor(1., batch_size * nb_imputation,  requires_grad=False).uniform_(0, 1) + 1e-9
      

        # USUALLY WE CAN GET V FROM U TO COUPLE BOTH OF THEM :
        u = (torch.rand(sample_shape, requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1)
        v = (torch.rand(sample_shape, requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1)
        if self.is_cuda:
            u = u.cuda()
            v = v.cuda()


        b = reparam_pz(u, pi_list_extended)
        z = Heaviside(b)
        tilde_b = reparam_pz_b(v, z, pi_list_extended)


        soft_concrete_rebar_z = sigma_lambda(b, torch.exp(self.temperature_total))  
        soft_concrete_rebar_tilde_z = sigma_lambda(tilde_b, torch.exp(self.temperature_total))

        # if not self.pytorch_relax :
        # else :
            # distrib = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.exp(self.temperature_total), probs = pi_list_extended)
            # soft_concrete_rebar_tilde_z = distrib.rsample()

        return tilde_b, soft_concrete_rebar_z, soft_concrete_rebar_tilde_z
