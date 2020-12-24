from .destructive_network import *
from .regularization import *


class DestructionModule():

    def __init__(self, destructor, regularization = None, destructorVar = None, regularization_var = None):
        self.destructor = destructor
        self.destructorVar = destructorVar
        self.regularization = regularization
        self.regularization_var = regularization_var


        if self.destructorVar is None :
            self.variational = False
        else :
            self.variational = True

    def train(self):
        self.destructor.train()
        if self.variational :
            self.destructorVar.train()

    def zero_grad(self):
        self.destructor.zero_grad()
        if self.variational :
            self.destructorVar.zero_grad()

                
    def eval(self):
        self.destructor.eval()
        if self.variational :
            self.destructorVar.eval()      
                


    def is_variational(self):
        return self.variational

    def cuda(self):
        self.destructor.cuda()
        if self.variational:
            self.destructorVar.cuda()

    def parameters(self):
        # if self.need_imputation and self. 
        # TODO Check learnable parameters of imputation method
        if self.variational :
            return [
                {'params': self.destructor.parameters()},
                {'params' : self.destructorVar.parameters()}
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
            pi_list_var = self.destructorVar(data_expanded, one_hot_target)
            if not test and self.regularization_var is not None :
                loss_reg_var = self.regularization_var(pi_list_var)
            else :
                loss_reg_var = torch.zeros((1))

            return pi_list_init, loss_reg, pi_list_var, loss_reg_var

        else :
            return pi_list_init, loss_reg, None, None