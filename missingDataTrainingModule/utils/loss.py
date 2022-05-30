import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn

from .utils import *


def define_target(data, index, target, one_hot_target, post_hoc = None, post_hoc_guidance = None, argmax_post_hoc = None, dim_output = None):
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




def calculate_cost(mask_expanded,
                    trainer,
                    data_expanded, # Shape is ( nb_sample_z_monte_carlo, batch_size, nb_sample_z_iwae, channel, dim...)
                    target_expanded,
                    one_hot_target_expanded,
                    dim_output,
                    index_expanded = None,
                    loss_function = None,
                    log_y_hat = None,
                    no_imputation = False,
                    ):


        if log_y_hat is None :
            mask_expanded = trainer.reshape(mask_expanded)
            if index_expanded is not None :
                index_expanded_flatten = index_expanded.flatten(0,2)
            else :
                index_expanded_flatten = None
            log_y_hat, _ = trainer.classification_module(data = data_expanded.flatten(0,2), mask = mask_expanded, index = index_expanded_flatten)


        nb_sample_z_monte_carlo, batch_size, nb_sample_z_iwae = data_expanded.shape[:3]
        if no_imputation :
            nb_imputation_iwae = 1
            nb_imputation_mc = 1
        else :
            if trainer.classification_module.training :
                nb_imputation_iwae = trainer.classification_module.imputation.nb_imputation_iwae
                nb_imputation_mc = trainer.classification_module.imputation.nb_imputation_mc
            else :
                nb_imputation_iwae = trainer.classification_module.imputation.nb_imputation_iwae_test
                nb_imputation_mc = trainer.classification_module.imputation.nb_imputation_mc_test
            

        target_expanded_flatten = target_expanded.flatten(0,2)
        if one_hot_target_expanded is not None :
            one_hot_target_expanded_flatten = one_hot_target_expanded.flatten(0,2)
            one_hot_target_expanded_multiple_imputation = extend_input(one_hot_target_expanded_flatten, mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
            one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,2)
        else :
            one_hot_target_expanded_multiple_imputation = None
            one_hot_target_expanded_multiple_imputation_flatten = None

        target_expanded_multiple_imputation = extend_input(target_expanded_flatten, mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
        loss_result = loss_function.eval(
                    input = log_y_hat,
                    target = target_expanded_multiple_imputation.flatten(0,2),
                    one_hot_target = one_hot_target_expanded_multiple_imputation_flatten,
                    iwae_mask = nb_sample_z_iwae,
                    iwae_sample = nb_imputation_iwae)

        

        loss_result = loss_result.reshape(nb_imputation_mc, nb_sample_z_monte_carlo, batch_size,)
        loss_result = loss_result.mean(dim = 0) # Mean on the mc imputation part



        return loss_result
  


def test_train_loss(trainer, loader, loss_function = None, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, mask_sampling = None,):
    """
    Evaluate trainer on the test set from loader using the given loss function.
    """
    trainer.train()
    if hasattr(trainer, "distribution_module") :
        trainer.distribution_module.eval()
    loss_train_total = 0.
    with torch.no_grad():
        for batch_index, data in enumerate(loader.test_loader):
            data, target, index = parse_batch(data)
            batch_size = data.shape[0]
            if trainer.use_cuda :
                data, target, index = on_cuda(data, target = target, index = index,)
            one_hot_target = get_one_hot(target, num_classes = loader.dataset.get_dim_output())
            target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = trainer.post_hoc, post_hoc_guidance = trainer.post_hoc_guidance, argmax_post_hoc = trainer.argmax_post_hoc, dim_output= loader.dataset.get_dim_output(),)
            
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

            if mask_sampling is None :
                z = None
                no_imputation = True
            else :
                z = mask_sampling(data = data, target = target, index = index, dataset = loader.dataset, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae)
                z = trainer.reshape(z)
                no_imputation = False


            log_y_hat, _ = trainer.classification_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
            log_y_hat = log_y_hat.reshape(-1, loader.dataset.get_dim_output())

            loss_train = calculate_cost(
                trainer = trainer,
                mask_expanded = z,
                data_expanded = data_expanded,
                target_expanded = target_expanded,
                index_expanded = index_expanded,
                one_hot_target_expanded = one_hot_target_expanded,
                dim_output = loader.dataset.get_dim_output(),
                log_y_hat = log_y_hat,
                loss_function = loss_function,
                no_imputation = no_imputation,
                )
            loss_train_total += loss_train.mean(0).sum(0) #Mean in MC sum in batch

    loss_train_total = loss_train_total / len(loader.test_loader.dataset)
    dic = {"train_loss_in_test": loss_train_total.item()}

    print('\nTest set: Train Loss in Test {:.4f}'.format(
        loss_train_total.item()))
            
    return dic


def multiple_test(trainer, loader, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1, mask_sampling = None, set_manual_seed = None,):
        """
        Evaluate accuracy, likelihood and mse of trainer on the test set from loader.

        Args:
            trainer (Trainer): The trainer to evaluate.
            loader (DataLoader): The data loader to use.
            nb_sample_z_monte_carlo (int): The number of Monte Carlo samples for the mask sampling if mask_sampling = True
            nb_sample_z_iwae (int): The number of IWAE samples for the mask sampling if mask_sampling = True
            mask_sampling (function): The function to use to sample the mask.
            set_manual_seed (int): Manual seed to use, useful for the evaluation of trainer when the mask can change a lot.
        """
        trainer.eval()
        mse_loss_selection = 0
        mse_loss_selection_prod = 0
        neg_likelihood_selection = 0
        correct_selection = 0
        dim_output = loader.dataset.get_dim_output()
        y_true = []
        y_pred = []
        
        if set_manual_seed is not None :
            print(f"SET MANUAL SEED {set_manual_seed}")
            torch.manual_seed(set_manual_seed)
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                batch_size = data.shape[0]
                if trainer.use_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)
                one_hot_target = get_one_hot(target, num_classes = dim_output)
                target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = trainer.post_hoc, post_hoc_guidance = trainer.post_hoc_guidance, argmax_post_hoc = trainer.argmax_post_hoc, dim_output= loader.dataset.get_dim_output(),)
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

                if mask_sampling is None :
                    z = None
                    no_imputation = True
                else :
                    z = mask_sampling(data, index, target, loader.dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae)
                    z = trainer.reshape(z)
                    no_imputation = False

                log_y_hat, _ = trainer.classification_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
                log_y_hat = log_y_hat.reshape(-1, dim_output)



                if torch.any(torch.isnan(log_y_hat)) :
                    print(torch.any(torch.isnan(z)))
                    assert 1==0


                
                mse_loss = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = dim_output,
                    loss_function=MSELossLastDim(reduction = 'none'),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                mse_loss_selection += mse_loss.mean(0).sum(0) # Mean in MC sum in batch


                mse_loss_prod = calculate_cost(
                    trainer = trainer,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = dim_output,
                    loss_function=MSELossLastDim(reduction = 'none', iwae_reg='prod'),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                mse_loss_selection_prod += mse_loss_prod.mean(0).sum(0) # Mean in MC sum in batch



                if dim_output > 1 :
                    if trainer.post_hoc and (not trainer.argmax_post_hoc) :
                        nll_loss = continuous_NLLLoss(reduction='none')
                    else :
                        nll_loss = NLLLossAugmented(reduction='none')
                    
                    nb_imputation_mc = trainer.classification_module.imputation.nb_imputation_mc_test
                    nb_imputation_iwae = trainer.classification_module.imputation.nb_imputation_iwae_test
                    target_expanded_multiple = target_expanded.reshape(nb_sample_z_monte_carlo * batch_size, nb_sample_z_iwae)[:,0].unsqueeze(0).expand(nb_imputation_mc, -1).flatten()
                    current_pred = log_y_hat.reshape(nb_imputation_mc * nb_sample_z_monte_carlo* batch_size, nb_sample_z_iwae * nb_imputation_iwae, dim_output).mean(-2).argmax(-1) 
                    y_true = y_true + target_expanded_multiple.cpu().tolist()
                    y_pred = y_pred + current_pred.cpu().tolist()
                        
                    accuracy = calculate_cost(
                        trainer = trainer,
                        mask_expanded = z,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dim_output,
                        loss_function=AccuracyLoss(reduction = 'none'),
                        log_y_hat = log_y_hat,
                        no_imputation = no_imputation,
                    )
                    correct_selection += accuracy.mean(0).sum(0) # Mean in MC sum in batch

                    neg_likelihood = calculate_cost(
                        trainer = trainer,
                        mask_expanded = z,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dim_output,
                        loss_function=nll_loss,
                        log_y_hat = log_y_hat,
                        no_imputation = no_imputation,
                        )
                    neg_likelihood_selection += neg_likelihood.mean(0).sum(0) #Mean in MC sum in batch

        

        mse_loss_selection /= len(loader.test_loader.dataset) 
        mse_loss_selection_prod /= len(loader.test_loader.dataset)
        
        if mask_sampling is None :
            suffix = "no_selection"
        else :
            suffix = "selection_mc_{}_iwae_{}_imputemc_{}_imputeiwae_{}".format(nb_sample_z_monte_carlo, nb_sample_z_iwae,
                                                                trainer.classification_module.imputation.nb_imputation_mc_test,
                                                                trainer.classification_module.imputation.nb_imputation_iwae_test)

        if dim_output > 1 :
            neg_likelihood_selection /= len(loader.test_loader.dataset)
            accuracy_selection = correct_selection / len(loader.test_loader.dataset)
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            y_true = np.array(y_true)
            dic = dic_evaluation(accuracy = accuracy_selection.item(),
                            neg_likelihood = neg_likelihood_selection.item(),
                            mse = mse_loss_selection.item(),
                            suffix = suffix,
                            confusion_matrix=confusion_matrix,)
            print(confusion_matrix)
            
            print('\nTest {} set: MSE: {:.4f}, MSE_PROD: {:.4f}, Likelihood {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
                    suffix, mse_loss_selection.item(), mse_loss_selection_prod.item(), -neg_likelihood_selection.item(),
                    correct_selection.item(), len(loader.test_loader.dataset),100. * correct_selection.item() / len(loader.test_loader.dataset),
                    )
                )
        else :
            dic = dic_evaluation(accuracy = None,
                            neg_likelihood = None,
                            mse = mse_loss_selection.item(),
                            suffix = suffix,
                            confusion_matrix=None,
                        )
            
            print('\nTest {} set: MSE: {:.4f}, MSE_PROD: {:.4f}, Likelihood {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
                    suffix, mse_loss_selection.item(), mse_loss_selection_prod.item(), -1.,
                    -1., len(loader.test_loader.dataset), -1.,
                    )
                )



      



        return dic





def eval_selection(trainer, loader,):
    trainer.classification_module.imputation.nb_imputation_mc_test = 1
    trainer.classification_module.imputation.nb_imputation_iwae_test = 1     
    trainer.eval()
    dic = eval_selection_local(trainer, loader,)
    if hasattr(trainer.distribution_module.distribution, "keywords") :
        if "rate" in trainer.distribution_module.distribution.keywords :
            rate = trainer.distribution_module.distribution.keywords["rate"]
        elif "k" in trainer.distribution_module.distribution.keywords :
            rate = trainer.distribution_module.distribution.keywords["k"] / np.prod(loader.dataset.get_dim_input(), dtype=np.float32)
    elif hasattr(trainer.selection_module, "regularization") and hasattr(trainer.selection_module.regularization, "rate"):
        rate = trainer.selection_module.regularization.rate
    else :
        rate = 0.
    if rate > 0.:
        aux_dic_rate = eval_selection_local(trainer, loader, rate)
        for key in aux_dic_rate.keys():
            dic[key+"_rate"] = aux_dic_rate[key]

    return dic



def eval_selection_local(trainer, loader, rate = None):
    dic = {}
    dic["fp"] = 0
    dic["tp"] = 0
    dic["fn"] = 0
    dic["tn"] = 0
    dic["selection_auroc"] = 0
    sum_error_round = 0
    sum_error = 0
    error_selection_auroc = False
    for batch in loader.test_loader :
        try :
            data, target, index = batch
        except :
            print("Should give index to get the eval selection")
            return {}
        if not hasattr(loader.dataset, "optimal_S_test") :
            raise AttributeError("This dataset do not have an optimal S defined")
        else :
            optimal_S_test = loader.dataset.optimal_S_test[index]

        X_test = data.type(torch.float32)
        Y_test = target.type(torch.float32)
        if trainer.use_cuda :
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
            optimal_S_test = optimal_S_test.cuda()

        if hasattr(trainer, "selection_module"):
            selection_module = trainer.selection_module
            distribution_module = trainer.distribution_module
            selection_module.eval()
            distribution_module.eval()
            selection_evaluation = True
        else :
            selection_evaluation = False
        classification_module = trainer.classification_module
        classification_module.eval()


        with torch.no_grad():
            log_pi_list, _ = selection_module(X_test)
            distribution_module(torch.exp(log_pi_list))
            z = distribution_module.sample((1,))
            z = trainer.reshape(z)
            if isinstance(selection_module.activation, torch.nn.LogSoftmax):
                pi_list = distribution_module.sample((100,)).mean(dim = 0)
            else :
                pi_list = torch.exp(log_pi_list)



        optimal_S_test = optimal_S_test.detach().cpu().numpy()
        if rate is None :
            sel_pred = (pi_list >0.5).detach().cpu().numpy().astype(int).reshape(-1)
        else :
            dim_pi_list = np.prod(trainer.selection_module.selector.output_size)
            k = max(int(dim_pi_list * rate),1)
            to_sel = torch.topk(pi_list.flatten(1), k, dim = 1)[1]
            sel_pred = torch.zeros_like(pi_list).scatter(1, to_sel, 1).detach().cpu().numpy().astype(int).reshape(-1)

        sel_true = optimal_S_test.reshape(-1)
        fp = np.sum((sel_pred == 1) & (sel_true == 0))
        tp = np.sum((sel_pred == 1) & (sel_true == 1))

        fn = np.sum((sel_pred == 0) & (sel_true == 1))
        tn = np.sum((sel_pred == 0) & (sel_true == 0))

       
        dic["fp"] += fp
        dic["tp"] += tp
        dic["fn"] += fn
        dic["tn"] += tn

        sum_error_round += np.sum(np.abs(optimal_S_test.reshape(-1) - sel_pred))
        sum_error += np.sum(np.abs(optimal_S_test.reshape(-1) - pi_list.reshape(-1).detach().cpu().numpy()))
        try :
            dic["selection_auroc"] += sklearn.metrics.roc_auc_score(optimal_S_test.reshape(-1), pi_list.reshape(-1))
        except :
            error_selection_auroc = True

    total = dic["tp"] + dic["tn"] + dic["fp"] + dic["fn"]
    dic["fpr"] = dic["fp"] / (dic["fp"] + dic["tn"] + 1e-8)
    dic["tpr"] = dic["tp"] / (dic["tp"] + dic["fn"] + 1e-8)
    dic["selection_accuracy_rounded"] = sum_error_round / total
    dic["selection_accuracy"] = sum_error / total

    if error_selection_auroc :
        dic["selection_auroc"] = -1
    else :
        dic["selection_auroc"] /= len(loader.test_loader)

    if rate is not None :
        print("Selection Test rate {:.3f} : fpr {:.4f} tpr {:.4f} auroc {:.4f} accuracy {:.4f}".format(rate, dic["fpr"], dic["tpr"], dic["selection_auroc"], dic["selection_accuracy"]))
    else :
        print("Selection Test : fpr {:.4f} tpr {:.4f} auroc {:.4f} accuracy {:.4f}".format(dic["fpr"], dic["tpr"], dic["selection_auroc"], dic["selection_accuracy"]))
    
    return dic