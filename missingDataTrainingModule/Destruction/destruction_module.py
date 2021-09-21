import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *
from collections.abc import Iterable

from .destructive_network import *
from .regularization import *



class DestructionModule():

    def __init__(self, destructor, regularization = None, feature_extractor = None, use_cuda = False):
        self.destructor = destructor
        self.regularization = regularization
        self.use_cuda = use_cuda

        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_extraction = False
        else :
            self.need_extraction = True


        if not self.regularization is list and self.regularization is not None:
           self.regularization = [self.regularization]
        # if self.use_cuda :
        #     for regul in self.regularization :
        #         regul = regul.cuda()


    def kernel_update(self, kernel_patch, stride_patch):
        self.destructor.kernel_update(kernel_patch, stride_patch)

    def train(self):
        self.destructor.train()

    def zero_grad(self):
        self.destructor.zero_grad()


                
    def eval(self):
        self.destructor.eval()
   
                



    def cuda(self):
        self.destructor = self.destructor.cuda()


    def parameters(self):
        return self.destructor.parameters()



    def __call__(self, data_expanded, one_hot_target = None, test=False):
        if self.need_extraction :
            data_expanded = self.feature_extractor(data_expanded)

        if one_hot_target is not None :
            log_pi_list = self.destructor(data_expanded, one_hot_target)
            loss_reg = torch.zeros((1)).cuda()
            if not test and self.regularization is not None :
                for reg in self.regularization :

                    print(torch.exp(log_pi_list[0]))
                    log_pi_list, loss_reg_aux = reg(log_pi_list)
                    print(torch.exp(log_pi_list[0]))
                    loss_reg +=loss_reg_aux
                    

            return log_pi_list, loss_reg

        else :
            log_pi_list = self.destructor(data_expanded)
            
            loss_reg = torch.zeros((1))
            if self.use_cuda :
                loss_reg = loss_reg.cuda()
            
            if not test and self.regularization is not None :
                for reg in self.regularization :
                    log_pi_list, loss_reg_aux = reg(log_pi_list)
                    loss_reg +=loss_reg_aux
                    
            return log_pi_list, loss_reg