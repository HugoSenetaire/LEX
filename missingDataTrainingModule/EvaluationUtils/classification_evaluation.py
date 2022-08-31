import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn

from ..utils.utils import *
from .loss import *
from .define_target import define_target
from .calculate_cost import calculate_cost



def test_train_loss(interpretable_module, loader, loss_function = None, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, mask_sampling = None, trainer = None):
    """
    Evaluate interpretable_module on the test set from loader using the given loss function.
    """

    if trainer is None :
        post_hoc = False
        post_hoc_guidance = None
        argmax_post_hoc = False
    else :
        post_hoc = trainer.post_hoc
        post_hoc_guidance = trainer.post_hoc_guidance
        argmax_post_hoc = trainer.argmax_post_hoc

    interpretable_module.train()
    if hasattr(interpretable_module, "distribution_module") :
        interpretable_module.distribution_module.eval()
    loss_train_total = 0.
    with torch.no_grad():
        for batch_index, data in enumerate(loader.test_loader):
            data, target, index = parse_batch(data)
            batch_size = data.shape[0]
            if next(interpretable_module.parameters()).is_cuda:
                data, target, index = on_cuda(data, target = target, index = index,)
            if trainer is not None :
                target = define_target(data,
                                        index,
                                        target,
                                        dim_output= loader.dataset.get_dim_output(),
                                        post_hoc = post_hoc,
                                        post_hoc_guidance = post_hoc_guidance,
                                        argmax_post_hoc = argmax_post_hoc,
                                        )

            
            data_expanded, target_expanded, index_expanded = sampling_augmentation(data,
                                                                            target = target,
                                                                            index=index,
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
                z = mask_sampling(data = data, index = index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae)
                z = interpretable_module.reshape(z)
                no_imputation = False


            log_y_hat, _ = interpretable_module.prediction_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
            if type(loader.dataset.get_dim_output()) == int :
                log_y_hat = log_y_hat.reshape(-1, loader.dataset.get_dim_output())
            else :
                log_y_hat = log_y_hat.reshape(-1, *loader.dataset.get_dim_output())

            loss_train = calculate_cost(
                interpretable_module = interpretable_module,
                mask_expanded = z,
                data_expanded = data_expanded,
                target_expanded = target_expanded,
                index_expanded = index_expanded,
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


def multiple_test_classification(interpretable_module, loader, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1, mask_sampling = None, set_manual_seed = None, trainer = None, prefix = ""):
        """
        Evaluate accuracy, likelihood and mse of interpretable_module on the test set from loader.

        Args:
            interpretable_module (interpretable_module): The interpretable_module to evaluate.
            loader (DataLoader): The data loader to use.
            nb_sample_z_monte_carlo (int): The number of Monte Carlo samples for the mask sampling if mask_sampling = True
            nb_sample_z_iwae (int): The number of IWAE samples for the mask sampling if mask_sampling = True
            mask_sampling (function): The function to use to sample the mask.
            set_manual_seed (int): Manual seed to use, useful for the evaluation of interpretable_module when the mask can change a lot.
        """
        interpretable_module.eval()
        mse_loss_selection = 0
        mse_loss_selection_prod = 0
        neg_likelihood_selection = 0
        correct_selection = 0
        dim_output = loader.dataset.get_dim_output()
        y_true = []
        y_pred = []

        if trainer is None :
            post_hoc = False
            post_hoc_guidance = None
            argmax_post_hoc = False
        else :
            post_hoc = trainer.post_hoc
            post_hoc_guidance = trainer.post_hoc_guidance
            argmax_post_hoc = trainer.argmax_post_hoc

    
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                batch_size = data.shape[0]
                if next(interpretable_module.parameters()).is_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)

                target = define_target(data,
                                        index,
                                        target,
                                        dim_output= loader.dataset.get_dim_output(),
                                        post_hoc = post_hoc,
                                        post_hoc_guidance = post_hoc_guidance,
                                        argmax_post_hoc = argmax_post_hoc,
                                        )

                
                data_expanded, target_expanded, index_expanded = sampling_augmentation(data,
                                                                target = target,
                                                                index=index,
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
                    z = mask_sampling(data, index, nb_sample_z_monte_carlo, nb_sample_z_iwae)
                    z = interpretable_module.reshape(z)
                    no_imputation = False

                log_y_hat, _ = interpretable_module.prediction_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
                if type(dim_output) == int :
                    log_y_hat = log_y_hat.reshape(-1, 1)
                else :
                    log_y_hat = log_y_hat.reshape(-1, *dim_output)



                if torch.any(torch.isnan(log_y_hat)) :
                    print(torch.any(torch.isnan(z)))
                    assert 1==0


                
                mse_loss = calculate_cost(
                    interpretable_module = interpretable_module,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dim_output,
                    loss_function=BrierScore(reduction = 'none', iwae_reg='mean'),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                mse_loss_selection += mse_loss.mean(0).sum(0) # Mean in MC sum in batch


                mse_loss_prod = calculate_cost(
                    interpretable_module = interpretable_module,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dim_output,
                    loss_function=BrierScore(reduction = 'none', iwae_reg='prod'),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                mse_loss_selection_prod += mse_loss_prod.mean(0).sum(0) # Mean in MC sum in batch



                
                if post_hoc and (not argmax_post_hoc) :
                    nll_loss = continuous_NLLLoss(reduction='none')
                else :
                    nll_loss = NLLLossAugmented(reduction='none')
                
                nb_imputation_mc = interpretable_module.prediction_module.imputation.nb_imputation_mc_test
                nb_imputation_iwae = interpretable_module.prediction_module.imputation.nb_imputation_iwae_test
                target_expanded_multiple = target_expanded.reshape(nb_sample_z_monte_carlo * batch_size, nb_sample_z_iwae)[:,0].unsqueeze(0).expand(nb_imputation_mc, -1).flatten()
                current_pred = log_y_hat.reshape(nb_imputation_mc * nb_sample_z_monte_carlo* batch_size, nb_sample_z_iwae * nb_imputation_iwae, dim_output).mean(-2).argmax(-1) 
                y_true = y_true + target_expanded_multiple.cpu().tolist()
                y_pred = y_pred + current_pred.cpu().tolist()
                    
                accuracy = calculate_cost(
                    interpretable_module = interpretable_module,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dim_output,
                    loss_function=AccuracyLoss(reduction = 'none'),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                correct_selection += accuracy.mean(0).sum(0) # Mean in MC sum in batch

                neg_likelihood = calculate_cost(
                    interpretable_module = interpretable_module,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dim_output,
                    loss_function=nll_loss,
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                    )
                neg_likelihood_selection += neg_likelihood.mean(0).sum(0) #Mean in MC sum in batch

        

        mse_loss_selection /= len(loader.test_loader.dataset) 
        mse_loss_selection_prod /= len(loader.test_loader.dataset)
        
        if mask_sampling is None :
            suffix = "{}no_selection".format(prefix)
        else :
            suffix = "{}selection_mc_{}_iwae_{}_imputemc_{}_imputeiwae_{}".format(prefix, nb_sample_z_monte_carlo, nb_sample_z_iwae,
                                                                interpretable_module.prediction_module.imputation.nb_imputation_mc_test,
                                                                interpretable_module.prediction_module.imputation.nb_imputation_iwae_test)

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

        print("============")
        print("\n")
        return dic



def multiple_test_regression(interpretable_module, loader, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1, mask_sampling = None, set_manual_seed = None, trainer = None, prefix = ""):
        """
        Evaluate mse of interpretable_module on the test set from loader.

        Args:
            interpretable_module (interpretable_module): The interpretable_module to evaluate.
            loader (DataLoader): The data loader to use.
            nb_sample_z_monte_carlo (int): The number of Monte Carlo samples for the mask sampling if mask_sampling = True
            nb_sample_z_iwae (int): The number of IWAE samples for the mask sampling if mask_sampling = True
            mask_sampling (function): The function to use to sample the mask.
            set_manual_seed (int): Manual seed to use, useful for the evaluation of interpretable_module when the mask can change a lot.
        """
        interpretable_module.eval()
        mse_loss_selection = 0
        dim_output = loader.dataset.get_dim_output()

        if trainer is None :
            post_hoc = False
            post_hoc_guidance = None
            argmax_post_hoc = False
        else :
            post_hoc = trainer.post_hoc
            post_hoc_guidance = trainer.post_hoc_guidance
            argmax_post_hoc = trainer.argmax_post_hoc

        

        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                if next(interpretable_module.parameters()).is_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)

                target = define_target(data,
                                        index,
                                        target,
                                        dim_output= loader.dataset.get_dim_output(),
                                        post_hoc = post_hoc,
                                        post_hoc_guidance = post_hoc_guidance,
                                        argmax_post_hoc = argmax_post_hoc,
                                        )

                
                data_expanded, target_expanded, index_expanded = sampling_augmentation(data,
                                                                        target = target,
                                                                        index=index,
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
                    z = mask_sampling(data, index, nb_sample_z_monte_carlo, nb_sample_z_iwae)
                    z = interpretable_module.reshape(z)
                    no_imputation = False

                log_y_hat, _ = interpretable_module.prediction_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten)
                if type(loader.dataset.get_dim_output())==int :
                    log_y_hat = log_y_hat.reshape(-1, dim_output)
                else :
                    log_y_hat = log_y_hat.reshape(-1, *loader.dataset.get_dim_output())



                if torch.any(torch.isnan(log_y_hat)) :
                    print(torch.any(torch.isnan(z)))
                    assert 1==0
                
                mse_loss = calculate_cost(
                    interpretable_module = interpretable_module,
                    mask_expanded = z,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dim_output,
                    loss_function=MSE_Regression(reduction = 'none',),
                    log_y_hat = log_y_hat,
                    no_imputation = no_imputation,
                )
                mse_loss_selection += mse_loss.mean(0).sum(0) # Mean in MC sum in batch


         
        mse_loss_selection /= len(loader.test_loader.dataset) 
        
        if mask_sampling is None :
            suffix = "{}no_selection".format(prefix)
        else :
            suffix = "{}selection_mc_{}_iwae_{}_imputemc_{}_imputeiwae_{}".format(prefix, nb_sample_z_monte_carlo, nb_sample_z_iwae,
                                                                interpretable_module.prediction_module.imputation.nb_imputation_mc_test,
                                                                interpretable_module.prediction_module.imputation.nb_imputation_iwae_test)

            dic = dic_evaluation(accuracy = None,
                            neg_likelihood = None,
                            mse = mse_loss_selection.item(),
                            suffix = suffix,
                            confusion_matrix=None,
                        )
            
            print('\nTest {} set: MSE: {:.4f}'.format(
                    suffix, mse_loss_selection.item(),)
                )

        print("============")
        print("\n")
        return dic



