import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn




class NLLLossAugmented():
    def __init__(self, weight = None, ignore_index = -100, reduction = 'none', **kwargs):
        self.loss = nn.NLLLoss(weight = weight, ignore_index = ignore_index, reduction = 'none', **kwargs)
        self.iwae_type = "log"
        self.reduction = reduction
        
    def eval(self, input, target, one_hot_target, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the input (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target (batch_size * iwae_mask * iwae_sample)
            one_hot_target: the one hot target (batch_size * iwae_mask * iwae_sample, nb_category)
        """

        if one_hot_target is None :
            raise ValueError("One_hot_target is not defined")
        outloss = self.loss.forward(input, target)

        outloss = outloss.reshape((-1, iwae_mask, iwae_sample))
        outloss = torch.logsumexp(outloss, dim=-1) - torch.log(torch.tensor(iwae_sample, dtype=torch.float32))
        outloss = torch.logsumexp(outloss, dim=-1) - torch.log(torch.tensor(iwae_mask, dtype=torch.float32))

        if self.reduction == 'none' :
            return outloss
        elif self.reduction == 'mean' :
            return torch.mean(outloss)
        elif self.reduction == 'sum' :
            return torch.sum(outloss)
        else :
            raise AttributeError("Reduction not recognized")
        



class continuous_NLLLoss():
    """
    This class is used as an extension to handle continuous value for the NLLLoss
    """
    def __init__(self, reduction = 'mean', **kwargs):
        self.reduction = reduction
        self.iwae_type = "log"

    def eval(self, input, target, one_hot_target, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the the log probability of the class (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target (batch_size * iwae_mask * iwae_sample)
            one_hot_target: the one hot target (batch_size * iwae_mask * iwae_sample, nb_category)
        """
        if one_hot_target is None :
            raise ValueError("One_hot_target is not defined")

        nb_category = one_hot_target.shape[-1]
        input = input.reshape((-1,nb_category))
        outloss = - torch.sum(input * one_hot_target, -1)
        outloss = outloss.reshape((-1, iwae_mask, iwae_sample))
        outloss = torch.logsumexp(outloss, dim=-1) - torch.log(torch.tensor(iwae_sample, dtype=torch.float32))
        outloss = torch.logsumexp(outloss, dim=-1) - torch.log(torch.tensor(iwae_mask, dtype=torch.float32))

        if self.reduction == 'none' :
            return outloss
        elif self.reduction == 'mean' :
            return torch.mean(outloss)
        elif self.reduction == 'sum' :
            return torch.sum(outloss)
        else :
            raise AttributeError("Reduction not recognized")
        

class MSELossLastDim():
    """
    This class is used as an extension to handle continuous value for the MSELoss
    """
    def __init__(self, reduction = 'mean', need_exp = True, iwae_reg = 'mean', **kwargs):
        self.need_exp = need_exp
        self.iwae_reg = iwae_reg
        self.reduction = reduction

    def eval(self, input, target, one_hot_target, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the log probability of the class (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target (batch_size * iwae_mask * iwae_sample)
            one_hot_target: the one hot target (batch_size * iwae_mask * iwae_sample, nb_category)
        """

        if one_hot_target is None :
            # Handle the case of MSE
            current_target = target.reshape((-1, iwae_mask, iwae_sample, 1))[:,0,0,:]
            current_input = input.reshape((-1, iwae_mask, iwae_sample, 1))
        else :
            nb_category = one_hot_target.shape[-1]
            if nb_category == 1 :
                raise ValueError("One_hot_target is not defined correctly")
            current_input = input.reshape((-1, iwae_mask, iwae_sample, nb_category))
            current_target = one_hot_target.reshape((-1, iwae_mask, iwae_sample, nb_category))[:,0,0,:]
        #TODO @hhjs : Mean the after exp or before ?
        if self.iwae_reg == 'mean' :
            current_input = torch.exp(current_input)
            current_input = torch.mean(current_input, dim=2) # IWAE Sample 
            current_input = torch.mean(current_input, dim=1) # IWAE Mask
        elif self.iwae_reg == 'prod' :
            current_input = torch.mean(current_input, dim=2) # IWAE Sample 
            current_input = torch.mean(current_input, dim=1) # IWAE Mask
            current_input =  torch.exp(current_input)
        else :
            raise AttributeError("IWAE Reg not recognized")

       


        out_loss = torch.sum(torch.pow(current_input - current_target, 2), -1)



        if self.reduction == 'none' :
            return out_loss
        elif self.reduction == 'mean' :
            return out_loss.mean()
        elif self.reduction == 'sum' :
            return out_loss.sum()
        else :
            raise AttributeError("Reduction not recognized")



class AccuracyLoss():
    def __init__(self, reduction = 'mean', need_exp = True, **kwargs):
        self.reduction = reduction
        self.need_exp = need_exp
        self.iwae_type = "mean"

    def eval(self, input, target, one_hot_target, iwae_mask=1, iwae_sample=1):

        if one_hot_target is None :
            raise ValueError("One_hot_target is not defined")
        nb_category = one_hot_target.shape[-1]
        current_target = target.reshape((-1, iwae_mask, iwae_sample))
       

        if iwae_sample > 1 :
            assert current_target[0,0,0] == current_target[0,0,1] 
        if iwae_mask >1 :
            assert current_target[0,0,0] ==current_target[0,1,0] 

        current_target = current_target[:,0,0]
        current_input = input.reshape((-1, iwae_mask, iwae_sample, nb_category))
        current_input = current_input.mean(dim=2).mean(dim=1)
        accuracy = torch.argmax(current_input, dim=-1).eq(current_target).type(torch.float32)
        return accuracy


