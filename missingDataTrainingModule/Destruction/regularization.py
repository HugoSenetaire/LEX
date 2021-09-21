import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

import torch




def free_regularization(log_pi_list):
  loss_reg = torch.mean(torch.mean(torch.exp(log_pi_list),-1)).squeeze() 
  return log_pi_list, loss_reg

def squared_regularization(log_pi_list, missing_rate = 0.5):
  batch_size = log_pi_list.shape[0]
  channels = log_pi_list.shape[1]
  mean_pi_list = torch.mean(torch.exp(log_pi_list),-1)
  regularizing_vector = torch.ones_like(mean_pi_list) * missing_rate
  loss_reg = torch.mean((mean_pi_list - regularizing_vector)**2) # Not absolute or squared ? Intger or rate ?
  return log_pi_list, loss_reg


def softmax_regularization(log_pi_list, missing_rate = 0.5):
  if missing_rate>1.0 or missing_rate<0.0:
    raise ValueError("Need a missing rate betzeen 0 and 1.")
  
  batch_size = log_pi_list.shape[0]
  channels = log_pi_list.shape[1]
  k_selected = torch.tensor(min(max(channels*missing_rate,1),channels))
  # log_pi_list = torch.nn.functional.logsigmoid(torch.nn.functional.log_softmax(log_pi_list, dim = -1) + torch.log(k_selected))
  log_pi_list = torch.nn.functional.log_softmax(log_pi_list, dim = -1) + torch.log(k_selected)
  log_pi_list = torch.clamp(log_pi_list, max = 0.0)
  loss_reg = torch.tensor(0.)
  return log_pi_list, loss_reg