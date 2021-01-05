import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

from .destructive_network import *
from .regularization import *



class DestructionModule():

    def __init__(self, destructor, regularization = None, destructor_var = None, regularization_var = None):
        self.destructor = destructor
        self.destructor_var = destructor_var
        self.regularization = regularization
        self.regularization_var = regularization_var


        if self.destructor_var is None :
            self.variational = False
        else :
            self.variational = True

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
        pi_list_init = self.destructor(data_expanded)
        if not test and self.regularization is not None :
            loss_reg = self.regularization(pi_list_init)
        else :
            loss_reg = torch.zeros((1))
        
        if do_variational and not self.variational :
            raise Exception("Should have a variational network for variational training")
        elif do_variational and one_hot_target is None :
            raise Exception("Should give one hot target expanded when using variational training")


        if do_variational and one_hot_target is not None :
            pi_list_var = self.destructor_var(data_expanded, one_hot_target)
            if not test and self.regularization_var is not None :
                loss_reg_var = self.regularization_var(pi_list_var)
            else :
                loss_reg_var = torch.zeros((1))

            return pi_list_init, loss_reg, pi_list_var, loss_reg_var

        else :
            return pi_list_init, loss_reg, None, None