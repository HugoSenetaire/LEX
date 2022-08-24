import torch.nn as nn
import torch.nn.functional as F




from ..utils.utils import *


def calculate_cost(mask_expanded,
                interpretable_module,
                data_expanded, # Shape is ( nb_sample_z_monte_carlo, batch_size, nb_sample_z_iwae, channel, dim...)
                target_expanded,
                one_hot_target_expanded,
                dim_output,
                index_expanded = None,
                loss_function = None,
                log_y_hat = None,
                no_imputation = False,
                ):
        """
        Standardized loss calculation to handle IWAE and MC imputation and masks.
        This can be used for training or testing.
        """
        if mask_expanded is not None :
            print(mask_expanded.shape)
            print(mask_expanded.sum(dim=-1))
        if log_y_hat is None :
            mask_expanded = interpretable_module.reshape(mask_expanded)
            if index_expanded is not None :
                index_expanded_flatten = index_expanded.flatten(0,2)
            else :
                index_expanded_flatten = None
            log_y_hat, _ = interpretable_module.prediction_module(data = data_expanded.flatten(0,2), mask = mask_expanded, index = index_expanded_flatten)


        nb_sample_z_monte_carlo, batch_size, nb_sample_z_iwae = data_expanded.shape[:3]
        if no_imputation :
            nb_imputation_iwae = 1
            nb_imputation_mc = 1
        else :
            if interpretable_module.prediction_module.training :
                nb_imputation_iwae = interpretable_module.prediction_module.imputation.nb_imputation_iwae
                nb_imputation_mc = interpretable_module.prediction_module.imputation.nb_imputation_mc
            else :
                nb_imputation_iwae = interpretable_module.prediction_module.imputation.nb_imputation_iwae_test
                nb_imputation_mc = interpretable_module.prediction_module.imputation.nb_imputation_mc_test
            

        target_expanded_flatten = target_expanded.flatten(0,2)
        if one_hot_target_expanded is not None :
            one_hot_target_expanded_flatten = one_hot_target_expanded.flatten(0,2)
            one_hot_target_expanded_multiple_imputation = extend_input(one_hot_target_expanded_flatten, mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
            one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,2)
        else :
            one_hot_target_expanded_multiple_imputation = None
            one_hot_target_expanded_multiple_imputation_flatten = None

        target_expanded_multiple_imputation = extend_input(target_expanded_flatten, mc_part = nb_imputation_mc, iwae_part = nb_imputation_iwae)
        
        # The loss function should calculate average on the IWAE part for both imputation and masks.
        loss_result = loss_function.eval(
                    input = log_y_hat,
                    target = target_expanded_multiple_imputation.flatten(0,2),
                    one_hot_target = one_hot_target_expanded_multiple_imputation_flatten,
                    iwae_mask = nb_sample_z_iwae,
                    iwae_sample = nb_imputation_iwae)

        

        loss_result = loss_result.reshape(nb_imputation_mc, nb_sample_z_monte_carlo, batch_size,)
        loss_result = loss_result.mean(dim = 0) # Mean on the mc imputation part



        return loss_result
  


