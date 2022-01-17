import copy

import torch
import torch.nn.functional as F
from torch.distributions import *
from functools import partial


# def g_u_theta(pi_list, u):
#     """ g(u,\theta) in REBAR"""
#     return safe_log_prob(pi_list/(1-pi_list)) + safe_log_prob(u/(1-u))

# def g_tilde(pi_list, v, z):
#     """ \Tilde(g)(v,h(b),\theta) in REBAR"""
#     pi_list = torch.clamp(pi_list, 1e-7, 1-1e-7)
#     case1 = safe_log_prob(v/(1-v)/(1-pi_list) +1)
#     case0 = -safe_log_prob(v/(1-v)/pi_list +1)
#     return torch.where(z, case1, case0)

# def H(x):
#     # Heaviside function, 0 if x < 0 else 1
#     return torch.div(F.threshold(x, 0, 0), x)

# def sigma_lambda(z, lambda):
    # return F.sigmoid(z / lambda) + 1e-9 


def safe_log_prob(x, eps=1e-8):
    return torch.log(torch.clamp(x, eps, 1.0))    

def sigma_lambda(z, lambda_value):
  return F.sigmoid(z / lambda_value)

def binary_log_likelihood(y, log_y_hat):
    # standard LL for vectors of binary labels y and log predictions log_y_hat
    return (y * -F.softplus(-log_y_hat)) + (1 - y) * (-log_y_hat - F.softplus(-log_y_hat))

def Heaviside(x):
    return torch.heaviside(x.detach(), torch.tensor(0., device = x.device))
    # Heaviside function, 0 if x < 0 else 1
    # if x.is_cuda :
    #   output = torch.where(x<0,torch.zeros(x.shape).cuda(), torch.ones(x.shape).cuda())
    # else :
    #   output = torch.where(x<0,torch.zeros(x.shape), torch.ones(x.shape))
    # return output
    # return torch.div(F.threshold(x, 0, 0), x) # This does not work and deliver an error

def reparam_pz(u, pi_list):
    return (safe_log_prob(pi_list) - safe_log_prob(1 - pi_list)) + (safe_log_prob(u) - safe_log_prob(1 - u))


def reparam_pz_b(v, b, theta):
    # From Appendix C of the paper, this is the reparameterization of p(z|b) for the 
    # case where b ~ Bernoulli($\theta$). Returns z_squiggle, a Gumbel RV
    return(b * F.softplus(safe_log_prob(v) - safe_log_prob((1 - v) * (1 - theta)))) \
        + ((1 - b) * (-F.softplus(safe_log_prob(v) - safe_log_prob(v * (1 - theta)))))





# def u_to_v(pi_list, u, eps = 1e-8, force_same = False, s = None, v_prime = None):
    # """Convert u to tied randomness in v."""



    # u_prime = F.sigmoid(-safe_log_prob(pi_list/(1-pi_list)))  # g(u') = 0
    # if not force_same:
        # v = s*(u_prime+v_prime*(1-u_prime)) + (1-s)*v_prime*u_prime
        # print("HERE")
        # v = s * (safe_log_prob(v_prime/(1-v_prime)/(1-pi_list) +1)) + (1-s) * -(safe_log_prob(v_prime/(1-v_prime)/pi_list +1))
    # else :

        # v_1 = (u - u_prime) / torch.clamp(1 - u_prime, eps, 1)
        # v_1 = torch.clamp(v_1.clone(), 0, 1).detach()
        # v_1 = v_1.clone()*(1 - u_prime) + u_prime


        # v_0 = u / torch.clamp(u_prime, eps, 1)
        # v_0 = torch.clamp(v_0.clone(), 0, 1).detach()
        # v_0 = v_0.clone() * u_prime

        # v = u.clone()
        # v[(u > u_prime).detach()] = v_1[(u > u_prime).detach()]
        # v[(u <= u_prime).detach()] = v_0[(u <= u_prime).detach()]
        # v = torch.where(u>u_prime, v_1, v_0)
        # v[(u > u_prime).detach()] = v_1[(u > u_prime).detach()]
        # v[(u <= u_prime).detach()] = v_0[(u <= u_prime).detach()]
        # TODO: add pytorch check
        #v = tf.check_numerics(v, 'v sampling is not numerically stable.')
        # v = v + (-v + u).detach()  # v and u are the same up to numerical errors
    return v



def vectorize_dic(parameters, named_parameters, set_none_to_zero=False, skip_none=False):
    if set_none_to_zero:
        return torch.cat([parameters[name].flatten() if parameters[name] is not None else
                    torch.zeros(p.shape).flatten() for name,p in named_parameters])
    elif skip_none:
        return torch.cat([parameters[name].flatten() for name,_ in named_parameters if parameters[name] is not None])
    else:
        return torch.cat([parameters[name].flatten() for name,_ in named_parameters])

def vectorize_gradient(gradient, named_parameters, set_none_to_zero=False, skip_none=False):
    if set_none_to_zero:
        return torch.cat([g.flatten() if g is not None else 
                    torch.zeros(p.shape).flatten() for g,(name, p) in zip(gradient, named_parameters)])
    elif skip_none:
        return torch.cat([g.flatten() for g,(name, p) in zip(gradient, named_parameters) if g is not None])
    else:
        return torch.cat([g.flatten() for g,(name, p) in zip(gradient, named_parameters)])


def add_gradients(gradients1, gradients2):
    g_dic = {}
    for name in gradients1.keys():
        if gradients1[name] is None :
            g_dic[name] = gradients2[name]
        elif gradients2[name] is None :
            g_dic[name] = gradients1[name]
        else :
            g_dic[name] = gradients1[name] + gradients2[name]
    return g_dic
