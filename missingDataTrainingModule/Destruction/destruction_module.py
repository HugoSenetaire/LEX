import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *
from collections.abc import Iterable

from .destructive_network import *
from .regularization import *



class DestructionModule():

    def __init__(self, destructor, regularization = None, feature_extractor = None):
        self.destructor = destructor
        self.regularization = regularization


        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_extraction = False
        else :
            self.need_extraction = True


        if not self.regularization is list and self.regularization is not None:
           self.regularization = [self.regularization]


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
            pi_list = self.destructor(data_expanded, one_hot_target)
            loss_reg = torch.zeros((1)).cuda()
            if not test and self.regularization is not None :
                for reg in self.regularization :
                    loss_reg += reg(pi_list)

            return pi_list, loss_reg

        else :
            pi_list = self.destructor(data_expanded)
            loss_reg = torch.zeros((1)).cuda()
            if not test and self.regularization is not None :
                for reg in self.regularization :
                    loss_reg += reg(pi_list)

            return pi_list, loss_reg