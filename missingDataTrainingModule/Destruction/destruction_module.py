import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *
from collections.abc import Iterable

from .destructive_network import *
from .regularization import *



class DestructionModule():

    def __init__(self, destructor, regularization = None, destructor_var = None, regularization_var = None, feature_extractor = None):
        self.destructor = destructor
        self.destructor_var = destructor_var
        self.regularization = regularization
        self.regularization_var = regularization_var


        if self.destructor_var is None :
            self.variational = False
        else :
            self.variational = True


        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_extraction = False
        else :
            self.need_extraction = True


        if not self.regularization is list and self.regularization is not None:
           self.regularization = [self.regularization]

        if not self.regularization_var is list and self.regularization_var is not None:
           self.regularization_var = [self.regularization]


    def kernel_update(self, kernel_patch, stride_patch):
        self.destructor.kernel_update(kernel_patch, stride_patch)
        if self.variational :
            self.destructor_var.kernel_update(kernel_patch, stride_patch)

    def train(self):
        self.destructor.train()
        if self.variational :
            self.destructor_var.train()

    def zero_grad(self):
        self.destructor.zero_grad()
        if self.variational :
            self.destructor_var.zero_grad()

                
    def eval(self):
        self.destructor.eval()
        if self.variational :
            self.destructor_var.eval()      
                


    def is_variational(self):
        return self.variational

    def cuda(self):
        self.destructor = self.destructor.cuda()
        if self.variational:
            self.destructor_var = self.destructor_var.cuda()

    def parameters(self):
        # if self.need_imputation and self. 
        # TODO Check learnable parameters of imputation method
        if self.variational :
            return [
                {'params': self.destructor.parameters()},
                {'params' : self.destructor_var.parameters()}
                ]
        else :
            return self.destructor.parameters()



    def __call__(self, data_expanded, one_hot_target = None, do_variational= False, test=False):
        if self.need_extraction :
            data_expanded = self.feature_extractor(data_expanded)



            
        
        if do_variational and not self.variational :
            raise Exception("Should have a variational network for variational training")
        elif do_variational and one_hot_target is None :
            raise Exception("Should give one hot target expanded when using variational training")


        if one_hot_target is not None :
            pi_list_init = self.destructor(data_expanded, one_hot_target)
            loss_reg = torch.zeros((1)).cuda()
            if not test and self.regularization is not None :
                for reg in self.regularization :
                    loss_reg += reg(pi_list_init)

            return pi_list_init, loss_reg, None, None

        else :
            pi_list_init = self.destructor(data_expanded)
            loss_reg = torch.zeros((1)).cuda()
            if not test and self.regularization is not None :
                for reg in self.regularization :
                    loss_reg += reg(pi_list_init)

            return pi_list_init, loss_reg, None, None