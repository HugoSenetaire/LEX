import torch.nn as nn
import torch.nn.functional as F
import torch


def define_target(data, index, target, one_hot_target, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None, ):
    """
    The target depend on different parameter if you are posthoc, posthoc_guidance, argmax post_hoc and so on.
    This function lead to the true target.

    param:
        data: the data (batch_size, nb_channel,other dim...)
        index: the index of the data (batch_size, )
        log_y_hat: the log of the output of the classification (batch_size, nb_category)
        target: the target (batch_size, )
        one_hot_target: the one hot target (batch_size, nb_category)
        post_hoc: Either if we have to consider post hoc
        post_hoc_guidance: the post hoc guidance 
        argmax_post_hoc: the argmax of the post hoc (batch_size, )

    return:
        wanted_target : the true target (batch_size, )
        wanted_one_hot_target: the wanted one_hot_target (batch_size, nb_category)
    """

    if not post_hoc :
        wanted_target = target
        wanted_one_hot_target = one_hot_target.type(torch.float32)
        
    elif post_hoc_guidance is not None :
        log_probs, _ = post_hoc_guidance(data, index = index)
        log_probs = log_probs.detach()
        if argmax_post_hoc :
            wanted_target = torch.argmax(log_probs, -1)
            wanted_one_hot_target = torch.zeros_like(log_probs)
            wanted_one_hot_target.scatter_(-1, wanted_target.unsqueeze(-1), 1)
        else :
            wanted_target = log_probs #In that case, we need a special form a loss because Nll wont work as is.
            wanted_one_hot_target = log_probs
    
    else :
        raise AttributeError("No guidance provided for posthoc")    

    return wanted_target, wanted_one_hot_target


class NLLLossAugmented(nn.NLLLoss):
    def __init__(self, weight = None, ignore_index = -100, reduction = 'mean', **kwargs):
        super(NLLLossAugmented, self).__init__(weight = weight, ignore_index = ignore_index, reduction = reduction, **kwargs)

    def forward(self, input, target, one_hot_target = None):
        """
        This function is used to compute the negative log likelihood loss
        param:
            input: the input (batch_size, nb_category)
            target: the target (batch_size, )
        """
        return super(NLLLossAugmented, self).forward(input, target)
    



class continuous_NLLLoss(nn.NLLLoss):
    """
    This class is used as an extension to handle continuous value for the NLLLoss
    """
    def __init__(self, weight = None, ignore_index = -100, reduction = 'mean', **kwargs):
        super(continuous_NLLLoss, self).__init__(weight = weight, ignore_index = ignore_index, reduction = reduction, **kwargs)
    def forward(self, input, target, one_hot_target = None):
        """
        This function is used to compute the negative log likelihood loss.
        Input should be the log of the output of the classification.
        """
        neg_likelihood = - torch.sum(input * target, -1)
        if self.reduction == 'none' :
            return neg_likelihood
        elif self.reduction == 'mean' :
            return torch.mean(neg_likelihood)
        elif self.reduction == 'sum' :
            return torch.sum(neg_likelihood)
        else :
            raise AttributeError("Reduction not recognized")
        

class MSELossLastDim(nn.MSELoss):
    """
    This class is used as an extension to handle continuous value for the MSELoss
    """
    def __init__(self, reduction = 'mean', need_exp = True, **kwargs):
        super(MSELossLastDim, self).__init__(reduction = reduction, **kwargs)
        self.need_exp = need_exp
        self.need_one_hot = False

    def forward(self, input, target, one_hot_target):
        """
        This function is used to compute the loss
        """
        input_exp = torch.exp(input)
        mse = torch.sum(torch.pow(input_exp - one_hot_target, 2), -1)
        if self.reduction == 'none' :
            return mse
        elif self.reduction == 'mean' :
            return mse.mean()
        elif self.reduction == 'sum' :
            return mse.sum()
        else :
            raise AttributeError("Reduction not recognized")



