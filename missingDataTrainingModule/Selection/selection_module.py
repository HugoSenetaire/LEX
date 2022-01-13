import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *
from collections.abc import Iterable

from .selective_network import *
from .regularization_module import *
import torch
import torch.nn as nn

class SelectionModule(nn.Module):

    def __init__(self, selector, activation = torch.nn.LogSigmoid(), regularization = None,):
        super(SelectionModule, self).__init__()
        self.selector = selector
        self.regularization = regularization
        self.activation = activation

        if self.regularization is not None :
            if not self.regularization is list :
                self.regularization = [self.regularization]
            # self.regularization = nn.ModuleList(self.regularization)
        self.use_cuda = False
    
    def cuda(self,):
        super(SelectionModule, self).cuda()
        self.use_cuda = True

    def __call__(self, data_expanded, one_hot_target = None, test=False):
        
        loss_reg = torch.tensor(0.)
        if self.use_cuda :
            loss_reg = loss_reg.cuda()

        if one_hot_target is not None :
            log_pi_list = self.selector(data_expanded, one_hot_target)
        else :
            log_pi_list = self.selector(data_expanded)

        if self.activation is not None :
            log_pi_list = self.activation(log_pi_list)

        if self.regularization is not None :
            for reg in self.regularization :
                log_pi_list, loss_reg_aux = reg(log_pi_list)
                if self.use_cuda:
                    loss_reg_aux.cuda()
                loss_reg +=loss_reg_aux
                
            return log_pi_list, loss_reg