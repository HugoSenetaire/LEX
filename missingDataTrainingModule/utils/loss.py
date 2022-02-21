import torch.nn.functional as F
import torch

def calculate_neg_likelihood(data, index, log_y_hat, target, one_hot_target, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None,):
    """
    Calculate the negative log likelihood of the classification per element of the batch, no reduction is done. 
    The calculation depends on the mode (post hoc, argmax post hoc, or none)

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
        the negative log likelihood (batch_size, )
    """
    if not post_hoc:
        neg_likelihood = F.nll_loss(log_y_hat, target.flatten(), reduce = False)

    elif post_hoc_guidance is not None :
        out_y, _ = post_hoc_guidance(data, index = index)
        out_y = out_y.detach()
        if argmax_post_hoc :
            out_y = torch.argmax(out_y, -1)
            neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce = False)
        else :
            neg_likelihood = - torch.sum(torch.exp(out_y) * log_y_hat, -1)
    
    else :
        raise AttributeError("No guidance provided for posthoc")

    return neg_likelihood

def calculate_sse(data, index, log_y_hat, target, one_hot_target, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None,):
    """
    Calculate the sum squared error of the classification per element of the batch, no reduction is done. 
    The calculation depends on the mode (post hoc, argmax post hoc, or none)

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
        the sum squared error (batch_size, )
    """
    one_hot_target = one_hot_target.type(torch.float32)
    y_hat = torch.exp(log_y_hat)
    if not post_hoc:
        mse_loss = torch.sum(F.mse_loss(y_hat, one_hot_target, reduce = False), dim =-1)

    elif post_hoc_guidance is not None :
        log_probs, _ = post_hoc_guidance(data, index = index)
        log_probs = log_probs.detach()
        if argmax_post_hoc :
            max_idx = torch.argmax(log_probs, -1, keepdim = True)
            out_y = torch.zeros_like(log_probs)
            out_y.scatter_(-1, max_idx, 1)
            mse_loss = torch.sum(F.mse_loss(y_hat, out_y, reduce = False, ), dim =-1)
        else :
            out_y = torch.exp(log_probs)
            mse_loss = torch.sum(torch.pow(out_y - y_hat,2), -1)
    
    else :
        raise AttributeError("No guidance provided for posthoc")

    return mse_loss