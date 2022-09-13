import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn
import numpy as np

# Regression loss extended for iwae handling :
class MSE_Regression():
    """
    This class is used as an extension to handle continuous value for the MSELoss
    """
    def __init__(self, reduction = 'mean', **kwargs):
        self.reduction = reduction

    def eval(self, input, target, dim_output, iwae_mask=1, iwae_sample=1,):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the output of each neural  (batch_size * iwae_mask * iwae_sample, dim_output)
            target: the target (batch_size * iwae_mask * iwae_sample, dim_output)
            dim_output: the number of category
        """
        current_input = input.reshape((-1, iwae_mask, iwae_sample, np.prod(dim_output)))
        current_target = target.reshape((-1, iwae_mask, iwae_sample, np.prod(dim_output)))[:,0,0,:]

        
        current_input = torch.mean(current_input, dim=2) # IWAE Sample 
        current_input = torch.mean(current_input, dim=1) # IWAE Mask
      

        out_loss = torch.sum(torch.pow(current_input - current_target, 2), -1)

        if self.reduction == 'none' :
            return out_loss
        elif self.reduction == 'mean' :
            return out_loss.mean()
        elif self.reduction == 'sum' :
            return out_loss.sum()
        else :
            raise AttributeError("Reduction not recognized")



# Classification Loss :
class NLLLossAugmented():
    def __init__(self, weight = None, ignore_index = -100, reduction = 'none', **kwargs):
        self.loss = nn.NLLLoss(weight = weight, ignore_index = ignore_index, reduction = 'none', **kwargs)
        self.reduction = reduction
        
    def eval(self, input, target, dim_output, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the input should be the log_probability of each class (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target (batch_size * iwae_mask * iwae_sample)
        """
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

    def eval(self, input, target, dim_output, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the the log probability of each class (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target given by another NN this should be probability, not log probability here(batch_size * iwae_mask * iwae_sample, nb_category)
        """

        input = input.reshape((-1,np.prod(dim_output)))
        target = target.reshape((-1,np.prod(dim_output)))
        assert input.shape == target.shape

        outloss = - torch.sum(input * target, -1)
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



class AccuracyLoss():
    def __init__(self, reduction = 'mean', **kwargs):
        self.reduction = reduction
        self.iwae_type = "mean"

    def eval(self, input, target, dim_output, iwae_mask=1, iwae_sample=1):

        nb_category = np.prod(dim_output)
        current_target = target.reshape((-1, iwae_mask, iwae_sample))
       

        if iwae_sample > 1 :
            assert current_target[0,0,0] == current_target[0,0,1] 
        if iwae_mask >1 :
            assert current_target[0,0,0] == current_target[0,1,0] 

        current_target = current_target[:,0,0]
        current_input = input.reshape((-1, iwae_mask, iwae_sample, nb_category))
        current_input = current_input.mean(dim=2).mean(dim=1)
        accuracy = torch.argmax(current_input, dim=-1).eq(current_target).type(torch.float32)
        return accuracy

        

class BrierScore():
    """
    This class is used as an extension to handle continuous value for the MSELoss
    """
    def __init__(self, reduction = 'mean', iwae_reg = 'mean', **kwargs):
        self.reduction = reduction
        self.iwae_reg = iwae_reg

    def eval(self, input, target, dim_output, iwae_mask=1, iwae_sample=1):
        """
        This function is used to compute the negative log likelihood loss while handling the iwae part of it.
        param:
            input: the log probability of the class (batch_size * iwae_mask * iwae_sample, nb_category)
            target: the target (batch_size * iwae_mask * iwae_sample, nb_category)
        """

        #TODO @hhjs : Mean the after exp or before ?
        
        assert type(dim_output) == int or len(dim_output) == 1
        

        current_input = input.reshape((-1,iwae_mask, iwae_sample, np.prod(dim_output)))
        batch_size = input.shape[0]
        current_target= target.reshape((batch_size, iwae_mask, iwae_sample,-1))
        if current_target.shape[-1] == 1 :
            current_target = torch.nn.functional.one_hot(current_target.flatten(), np.prod(dim_output)).type(torch.float32)
        current_target = current_target.reshape((batch_size, iwae_mask, iwae_sample, np.prod(dim_output)))

        
        if iwae_sample > 1 :
            assert torch.all(current_target[0,0,0] == current_target[0,0,1]) 
        if iwae_mask >1 :
            assert torch.all(current_target[0,0,0] == current_target[0,1,0]) 
        current_target = current_target[:,0,0,:]


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

        outloss = torch.sum(torch.pow(current_input - current_target, 2), -1)

        if self.reduction == 'none' :
            return outloss
        elif self.reduction == 'mean' :
            return torch.mean(outloss)
        elif self.reduction == 'sum' :
            return torch.sum(outloss)
        else :
            raise AttributeError("Reduction not recognized")