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



def regular_scheduler(temperature, epoch, cste = 0.999):
    return temperature * cste


def get_distribution_module_from_args(args_distribution_module):

    assert(args_distribution_module["distribution"] is not None)
    # if args_distribution_module["distribution_module"] is REBAR_Distribution :
        # assert(args_distribution_module["distribution_relaxed"] is not None)
    # print( args_distribution_module["distribution_module"])
    print(args_distribution_module)
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
        self.distribution = distribution
        self.current_distribution = distribution

    def forward(self, log_distribution_parameters,):
        self.current_distribution = self.distribution(torch.exp(log_distribution_parameters))

    def log_prob(self, x):
        return self.current_distribution.log_prob(x)

    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)


    def sample(self, sample_shape= (1,)):
        if self.antitheis_sampling and self.training:
            if sample_shape[-1] == 1 :
                raise(AttributeError("Antitheis sampling only works for nb_sample_z_monte_carlo > 1"))
            
            aux_sample_shape = torch.Size(sample_shape[:-1]) + torch.Size((sample_shape[-1] // 2,) )
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
    def __init__(self, distribution, temperature_init = 0.5, scheduler_parameter = regular_scheduler, test_temperature = 0.0, antitheis_sampling=False,  **kwargs):
        super(DistributionWithSchedulerParameter, self).__init__(distribution, antitheis_sampling= antitheis_sampling)
        self.current_distribution = None
        self.temperature_total = torch.tensor(temperature_init, dtype=torch.float32)
        self.temperature = torch.tensor(temperature_init, dtype=torch.float32)
        self.test_temperature = test_temperature
        self.scheduler_parameter = scheduler_parameter


    def forward(self, distribution_parameters):
        if self.training :
            # print(torch.exp(distribution_parameters))
            self.current_distribution = self.distribution(probs = torch.exp(distribution_parameters), temperature = self.temperature)
        else :
            self.current_distribution = self.distribution(probs = torch.exp(distribution_parameters), temperature = torch.tensor(0., dtype=torch.float32))
        return self.current_distribution

    # TODO : WHAT IS GOING ON HERE ?

    # def eval(self,):
        # super().eval()
        # self.temperature = self.test_temperature
    
    # def train(self,):
        # super().train()
        # self.temperature = self.temperature_total 

    def sample_function(self, sample_shape):
        if self.training :
            return self.current_distribution.rsample(sample_shape)
        else :
            return self.current_distribution.sample(sample_shape)

    def update_distribution(self, epoch = None):
        if self.scheduler_parameter is not None :
            self.temperature_total = self.scheduler_parameter(self.temperature_total, epoch)


class FixedBernoulli(DistributionModule):
    def __init__(self, **kwargs):
        super(FixedBernoulli, self).__init__(distribution = torch.distributions.Bernoulli)
    
    def forward(self, distribution_parameters, ):
        self.current_distribution = self.distribution(probs = torch.ones_like(distribution_parameters, dtype = torch.float32)*torch.tensor(0.5))
        return self.current_distribution
        
    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)



class REBAR_Distribution(DistributionModule):
    def __init__(self, distribution, distribution_relaxed, temperature_init = 1.0, trainable = False, antitheis_sampling = False, force_same = True, **kwargs):
        super(REBAR_Distribution, self).__init__(distribution, antitheis_sampling=antitheis_sampling)
        if self.antitheis_sampling :
            raise AttributeError("Antitheis sampling only works for regular distribution")
        self.distribution_relaxed = distribution_relaxed
        self.force_same = force_same
        self.trainable = trainable
        self.temperature_total = torch.nn.Parameter(torch.tensor(temperature_init), requires_grad = trainable)

    def forward(self, distribution_parameters):
        # print(torch.exp(distribution_parameters))
        self.current_distribution = self.distribution(probs = torch.exp(distribution_parameters),)
        self.current_distribution_relaxed = self.distribution_relaxed(probs = torch.exp(distribution_parameters), temperature = self.temperature_total )
        self.distribution_parameters = distribution_parameters
        return self.current_distribution, self.current_distribution_relaxed

    def sample(self, sample_shape= (1,)):
        if self.training :
            shape_distribution_parameters = self.distribution_parameters.shape
            complete_size_reshape = torch.Size([1 for i in range(len(sample_shape))]) + shape_distribution_parameters
            complete_size = torch.Size(sample_shape) + shape_distribution_parameters 

            log_pi_list_extended = self.distribution_parameters.reshape(complete_size_reshape).expand(complete_size)
            pi_list = torch.exp(log_pi_list_extended)
            wanted_device = self.distribution_parameters.device
            u = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            v_p = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            z = reparam_pz(u, pi_list)
            s = Heaviside(z)
            z_tilde = reparam_pz_b(v_p, s, pi_list)
            sig_z = sigma_lambda(z, self.temperature_total)
            sig_z_tilde = sigma_lambda(z_tilde, self.temperature_total)

            return [sig_z, s, sig_z_tilde]
        else :
            return self.current_distribution.sample(sample_shape)



class REBAR_Distribution_STE(DistributionModule):
    def __init__(self, distribution, distribution_relaxed, temperature_init = 1.0, trainable = False, antitheis_sampling = False, force_same = True, **kwargs):
        super(REBAR_Distribution_STE, self).__init__(distribution, antitheis_sampling=antitheis_sampling)
        if self.antitheis_sampling :
            raise AttributeError("Antitheis sampling only works for regular distribution")
        self.distribution_relaxed = distribution_relaxed
        self.force_same = force_same
        self.trainable = trainable
        self.temperature_total = torch.nn.Parameter(torch.tensor(temperature_init), requires_grad = trainable)

    def forward(self, distribution_parameters):
        self.current_distribution = self.distribution(probs = torch.exp(distribution_parameters),)
        self.current_distribution_relaxed = self.distribution_relaxed(probs = torch.exp(distribution_parameters), temperature = self.temperature_total )
        self.distribution_parameters = distribution_parameters
        return self.current_distribution, self.current_distribution_relaxed

    def sample(self, sample_shape= (1,)):
        
        if self.training :
            shape_distribution_parameters = self.distribution_parameters.shape
            complete_size_reshape = torch.Size([1 for i in range(len(sample_shape))]) + shape_distribution_parameters
            complete_size = torch.Size(sample_shape) + shape_distribution_parameters 

            log_pi_list_extended = self.distribution_parameters.reshape(complete_size_reshape).expand(complete_size)

            wanted_device = self.distribution_parameters.device
            u = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            v_p = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)

            z = reparam_pz(u, torch.exp(log_pi_list_extended))

            s = Heaviside(z)
            z_tilde = reparam_pz_b(v_p, s, torch.exp(log_pi_list_extended))
            sig_z = sigma_lambda(z, self.temperature_total)
            sig_z_tilde = sigma_lambda(z_tilde, self.temperature_total)
            sig_z = threshold_STE.apply(sigma_lambda(z, self.temperature_total), 0.5)
            sig_z_tilde = threshold_STE.apply(sigma_lambda(z_tilde, self.temperature_total), 0.5)

            return [sig_z, s, sig_z_tilde]
        else :
            return self.current_distribution.sample(sample_shape)

