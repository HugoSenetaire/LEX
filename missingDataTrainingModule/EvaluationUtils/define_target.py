import torch.nn as nn
import torch.nn.functional as F
import torch



def define_target(data, index, target, dim_output, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None, ):
    """
    The target depend on different parameter if you are posthoc, posthoc_guidance, argmax post_hoc and so on.
    This function lead to target used for the loss function.

    param:
        data: the data (batch_size, nb_channel,other dim...)
        index: the index of the data (batch_size, )
        log_y_hat: the log of the output of the classification (batch_size, nb_category)
        target: the target (batch_size, )
        post_hoc: Either if we have to consider post hoc
        post_hoc_guidance: the post hoc guidance 
        argmax_post_hoc: the argmax of the post hoc (batch_size, )

    return:
        wanted_target : the true target (batch_size, ) depending on whether one use the continuous NLL or other, this can also be in the shape (batch_size, nb_category)
    """
    if dim_output is None :
        raise ValueError("dim_output is not defined")

    if type(dim_output) == int or len(dim_output) == 1 : 
        if type(dim_output) == int :
            nb_category = dim_output
        else :
            nb_category = dim_output[0]
        if nb_category > 1 :
            if not post_hoc :
                wanted_target = target
            elif post_hoc_guidance is not None :
                log_probs, _ = post_hoc_guidance(data, index = index)
                log_probs = log_probs.detach()
                if argmax_post_hoc :
                    wanted_target = torch.argmax(log_probs, -1)
                else :
                    wanted_target = torch.exp(log_probs)
            else :
                raise AttributeError("No guidance provided for posthoc") 
        else :
            if not post_hoc :
                wanted_target = target
            else :
                output_post_hoc, _ = post_hoc_guidance(data, index = index)
                wanted_target = output_post_hoc.detach()
    else :
        wanted_target = target


    return wanted_target


