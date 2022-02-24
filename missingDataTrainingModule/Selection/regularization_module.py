from math import log
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch




class LossRegularization():
  def __init__(self, lambda_reg, regul_loss = "L1", batched = False, rate = 0.0, **kwargs):
    self.lambda_reg = lambda_reg
    self.regul_loss = regul_loss
    self.rate = rate
    if self.rate>1.0 or self.rate<0.0:
      raise ValueError("Need a missing rate between 0 and 1.")

    if self.regul_loss=="L1":
      self.function = lambda x: torch.abs(x)
    elif self.regul_loss=="L2":
      self.function = lambda x: torch.pow(x, 2)
    else :
      raise AttributeError("regul_loss must be L1 or L2")
    self.batched = batched
    
  def __call__(self, log_pi_list):
    if self.lambda_reg == 0:
      loss_reg = torch.tensor(0., device = log_pi_list.device)
      return log_pi_list, loss_reg
    pi_list = torch.exp(log_pi_list)
    if self.batched:
      pi_list = torch.mean(pi_list, -1)

    regularizing_vector = torch.full_like(pi_list, self.rate)
    loss_reg = self.lambda_reg * torch.mean(self.function(regularizing_vector - pi_list))
    return log_pi_list, loss_reg
      
class SoftmaxRegularization():
  def __init__(self, rate = 0.5, batched = False, **kwargs):
    self.rate =rate
    self.batched = batched
    if self.rate>1.0 or self.rate<0.0:
      raise ValueError("Need a missing rate between 0 and 1.")
   
  
  def __call__(self, log_pi_list):
      
    batch_size = log_pi_list.shape[0]
    nb_dim = log_pi_list.shape[1]


    if self.batched:
      select_among = batch_size*nb_dim
    else :
      select_among = nb_dim
      
    k_selected = torch.tensor(min(max(select_among*self.rate,1),select_among))
    if log_pi_list.is_cuda:
      k_selected = k_selected.cuda()
    
    if self.batched:
      log_pi_list = torch.nn.functional.log_softmax(log_pi_list) + torch.log(k_selected)
    else :
      log_pi_list = torch.nn.functional.log_softmax(log_pi_list, dim = -1) + torch.log(k_selected)
    
    
    log_pi_list = torch.clamp(log_pi_list, max = 0.0)

    loss_reg = torch.tensor(0.)
    if log_pi_list.is_cuda:
      loss_reg = loss_reg.cuda()

    return log_pi_list, loss_reg


