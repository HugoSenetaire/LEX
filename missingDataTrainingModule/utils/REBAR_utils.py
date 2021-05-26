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
    return torch.log(torch.clamp(x, eps, 1.0-1e-8))    

def sigma_lambda(z, lambda_value):
  return F.sigmoid(z / lambda_value)+1e-9

def binary_log_likelihood(y, log_y_hat):
    # standard LL for vectors of binary labels y and log predictions log_y_hat
    return (y * -F.softplus(-log_y_hat)) + (1 - y) * (-log_y_hat - F.softplus(-log_y_hat))

def Heaviside(x):
    # Heaviside function, 0 if x < 0 else 1
    if x.is_cuda :
      output = torch.where(x<0,torch.zeros(x.shape).cuda(), torch.ones(x.shape).cuda())
    else :
      output = torch.where(x<0,torch.zeros(x.shape), torch.ones(x.shape))
    return output

# def Heaviside(x):
#   return torch.div(F.threshold(x, 0, 0), x) => This is very stupid


def reparam_pz(u, theta):
    return (safe_log_prob(theta) - safe_log_prob(1 - theta)) + (safe_log_prob(u) - safe_log_prob(1 - u))

def reparam_pz_b(v, b, theta):
    # From Appendix C of the paper, this is the reparameterization of p(z|b) for the 
    # case where b ~ Bernoulli($\theta$). Returns z_squiggle, a Gumbel RV
    return(b * F.softplus(safe_log_prob(v) - safe_log_prob((1 - v) * (1 - theta)))) \
        + ((1 - b) * (-F.softplus(safe_log_prob(v) - safe_log_prob(v * (1 - theta)))))

# def reparam_pz_b(v, b, theta):
#     # From Appendix C of the paper, this is the reparameterization of p(z|b) for the 
#     # case where b ~ Bernoulli($\theta$). Returns z_squiggle, a Gumbel RV
#     return(b * safe_log_prob(v) - safe_log_prob((1 - v) * (1 - theta))) \
#         + ((1 - b) * (safe_log_prob(v) - safe_log_prob(v * (1 - theta))))

def u_to_v(pi_list, u, eps = 1e-8):
    """Convert u to tied randomness in v."""
    u_prime = F.sigmoid(safe_log_prob(pi_list/(1-pi_list)))  # g(u') = 0
    # print(u_prime.shape)
    # print(u.shape)
    v_1 = (u - u_prime) / torch.clamp(1 - u_prime, eps, 1)
    v_1 = torch.clamp(v_1.clone(), 0, 1).detach()
    v_1 = v_1.clone()*(1 - u_prime) + u_prime
    v_0 = u / torch.clamp(u_prime, eps, 1)
    v_0 = torch.clamp(v_0.clone(), 0, 1).detach()
    v_0 = v_0.clone() * u_prime
    v = u.clone()
    v[(u > u_prime).detach()] = v_1[(u > u_prime).detach()]
    v[(u <= u_prime).detach()] = v_0[(u <= u_prime).detach()]
    # TODO: add pytorch check
    #v = tf.check_numerics(v, 'v sampling is not numerically stable.')
    vv = v + (-v + u).detach()  # v and u are the same up to numerical errors
    return vv



