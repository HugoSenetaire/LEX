import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import *


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


        nb_category = one_hot_target.shape[-1]
        current_input = input.reshape((-1, iwae_mask, iwae_sample, nb_category))
        current_one_hot_target = one_hot_target.reshape((-1, iwae_mask, iwae_sample, nb_category))[:,0,0,:]
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
        
        out_loss = torch.sum(torch.pow(current_input - current_one_hot_target, 2), -1)

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
        nb_category = one_hot_target.shape[-1]
        current_target = target.reshape((-1, iwae_mask, iwae_sample))
        current_target = current_target[:,0,0]
        current_input = input.reshape((-1, iwae_mask, iwae_sample, nb_category))
        current_input = current_input.mean(dim=2).mean(dim=1)
        accuracy = torch.argmax(current_input, dim=-1).eq(current_target).type(torch.float32)
        return accuracy




def calculate_cost(mask_expanded,
                    trainer,
                    data_expanded, # Shape is (nb_imputation, nb_sample_z_monte_carlo, nb_sample_z_iwae, batch_size, channel, dim...)
                    target_expanded,
                    one_hot_target_expanded,
                    dim_output,
                    index_expanded = None,
                    loss_function = None,
                    log_y_hat = None,
                    ):
        if log_y_hat is None :
            mask_expanded = trainer.reshape(mask_expanded)
            if index_expanded is not None :
                index_expanded_flatten = index_expanded.flatten(0,2)
            else :
                index_expanded_flatten = None

            log_y_hat, _ = trainer.classification_module(data_expanded.flatten(0,2), mask_expanded, index = index_expanded_flatten)
        

        nb_sample_z_monte_carlo, batch_size, nb_sample_z_iwae = data_expanded.shape[:3]
        if trainer.classification_module.training :
            nb_imputation_iwae = trainer.classification_module.imputation.nb_imputation_iwae
            nb_imputation_mc = trainer.classification_module.imputation.nb_imputation_mc
        else :
            nb_imputation_iwae = trainer.classification_module.imputation.nb_imputation_iwae_test
            nb_imputation_mc = trainer.classification_module.imputation.nb_imputation_mc_test


        target_expanded_multiple_imputation = extend_input(target_expanded.flatten(0,2), mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
        one_hot_target_expanded_multiple_imputation = extend_input(one_hot_target_expanded.flatten(0,2), mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
        loss_result = loss_function.eval(
                    input = log_y_hat,
                    target = target_expanded_multiple_imputation.flatten(0,2),
                    one_hot_target = one_hot_target_expanded_multiple_imputation.flatten(0,2),
                    iwae_mask = nb_sample_z_iwae,
                    iwae_sample = nb_imputation_iwae)

        

        loss_result = loss_result.reshape(nb_imputation_mc, nb_sample_z_monte_carlo, batch_size,)
        loss_result = loss_result.mean(dim = 0) # Mean on the mc imputation part

        return loss_result
  

def test_selection(trainer, loader, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, mask_sampling = None,):
        trainer.eval()
        mse_loss_selection = 0
        mse_loss_selection_prod = 0
        neg_likelihood_selection = 0
        correct_selection = 0
        

        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                batch_size = data.shape[0]
                if trainer.use_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)
                one_hot_target = get_one_hot(target, num_classes = loader.dataset.get_dim_output())
                target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = trainer.post_hoc, post_hoc_guidance = trainer.post_hoc_guidance, argmax_post_hoc = trainer.argmax_post_hoc)
                
                data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                                target = target,
                                                                                                                index=index,
                                                                                                                one_hot_target = one_hot_target,
                                                                                                                mc_part = nb_sample_z_monte_carlo,
                                                                                                                iwae_part = nb_sample_z_iwae,
                                                                                                                )
                if index_expanded is not None :
                    index_expanded_flatten = index_expanded.flatten(0,2)
                else :
                    index_expanded_flatten = None


                z = mask_sampling(data, index, target, loader.dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae)
                z = trainer.reshape(z)



                log_y_hat, _ = trainer.classification_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
                log_y_hat = log_y_hat.reshape(-1, loader.dataset.get_dim_output())

                if trainer.post_hoc and (not trainer.argmax_post_hoc) :
                    nll_loss = continuous_NLLLoss(reduction='none')
                else :
                    nll_loss = NLLLossAugmented(reduction='none')

                neg_likelihood = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = loader.dataset.get_dim_output(),
                    loss_function=nll_loss,
                    log_y_hat = log_y_hat,
                    )
                neg_likelihood_selection += neg_likelihood.mean(0).sum(0) #Mean in MC sum in batch

                mse_loss = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = loader.dataset.get_dim_output(),
                    loss_function=MSELossLastDim(reduction = 'none'),
                    log_y_hat = log_y_hat,
                )
                mse_loss_selection += mse_loss.mean(0).sum(0) # Mean in MC sum in batch


                mse_loss_prod = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = loader.dataset.get_dim_output(),
                    loss_function=MSELossLastDim(reduction = 'none', iwae_reg='prod'),
                    log_y_hat = log_y_hat,
                )
                mse_loss_selection_prod += mse_loss.mean(0).sum(0) # Mean in MC sum in batch


                accuracy = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = loader.dataset.get_dim_output(),
                    loss_function=AccuracyLoss(reduction = 'none'),
                    log_y_hat = log_y_hat,
                )
                correct_selection += accuracy.mean(0).sum(0) # Mean in MC sum in batch


        mse_loss_selection /= len(loader.test_loader.dataset) 
        neg_likelihood_selection /= len(loader.test_loader.dataset)
        accuracy_selection = correct_selection / len(loader.test_loader.dataset)
        mse_loss_selection_prod /= len(loader.test_loader.dataset)
        suffix = "selection_mc_{}_iwae_{}_imputemc_{}_iwaemc_{}".format(nb_sample_z_monte_carlo, nb_sample_z_iwae,
                                                                trainer.classification_module.imputation.nb_imputation_mc_test,
                                                                trainer.classification_module.imputation.nb_imputation_iwae_test)
        dic = dic_evaluation(accuracy = accuracy_selection.item(),
                            neg_likelihood = neg_likelihood_selection.item(),
                            mse = mse_loss_selection.item(),
                            suffix = suffix,)


        print('\nTest {} set: MSE: {:.4f}, MSE_PROD: {:.4f}, Likelihood {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
                suffix, mse_loss_selection.item(), mse_loss_selection_prod.item(), -neg_likelihood_selection.item(),
                correct_selection.item(), len(loader.test_loader.dataset),100. * correct_selection.item() / len(loader.test_loader.dataset),
                )
            )

        return dic




def test_no_selection(trainer, loader, ):
    trainer.eval()
    mse_loss_no_selection = 0
    neg_likelihood_no_selection = 0
    correct_no_selection = 0
    
    with torch.no_grad():
        for batch_index, data in enumerate(loader.test_loader):
            data, target, index = parse_batch(data)
            batch_size = data.shape[0]
            if trainer.use_cuda :
                data, target, index = on_cuda(data, target = target, index = index,)
            one_hot_target = get_one_hot(target, num_classes = loader.dataset.get_dim_output())
            ## Check the prediction without selection on the baseline method
            log_y_hat, _ = trainer.classification_module(data, index = index)
            pred_no_selection = torch.argmax(log_y_hat,dim = 1)
            neg_likelihood_no_selection += F.nll_loss(log_y_hat, target, reduction = 'sum')
            mse_loss_no_selection += F.mse_loss(torch.exp(log_y_hat), one_hot_target, reduce=False).sum(-1).sum()
            correct_no_selection += pred_no_selection.eq(target).sum()
        
        mse_loss_no_selection /= len(loader.test_loader.dataset) 
        neg_likelihood_no_selection /= len(loader.test_loader.dataset)
        accuracy_no_selection = correct_no_selection / len(loader.test_loader.dataset)
        suffix = "no_selection"
        dic = dic_evaluation(accuracy = accuracy_no_selection.item(),
                            neg_likelihood = neg_likelihood_no_selection.item(),
                            mse = mse_loss_no_selection.item(),
                            suffix = suffix,)

        print('\nTest {} set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
                    suffix, mse_loss_no_selection, -neg_likelihood_no_selection,
                    correct_no_selection, len(loader.test_loader.dataset),100. * correct_no_selection / len(loader.test_loader.dataset),
                    )
            )

        return dic