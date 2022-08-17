import torch.nn as nn
import torch.nn.functional as F
import torch



def define_target(data, index, target, one_hot_target, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None, dim_output = None):
    """
    The target depend on different parameter if you are posthoc, posthoc_guidance, argmax post_hoc and so on.
    This function lead to target used for the loss function.

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
    if dim_output is None :
        raise ValueError("dim_output is not defined")

    if dim_output > 1 :
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
                #In that case, we need a special form a loss because Nll wont work as is.
                wanted_target = torch.argmax(log_probs, -1) 
                wanted_one_hot_target = torch.exp(log_probs)
        else :
            raise AttributeError("No guidance provided for posthoc") 
    else :
        wanted_one_hot_target = None
        if not post_hoc :
            wanted_target = target
        else :
            output_post_hoc, _ = post_hoc_guidance(data, index = index)
            wanted_target = output_post_hoc.detach()


    return wanted_target, wanted_one_hot_target


