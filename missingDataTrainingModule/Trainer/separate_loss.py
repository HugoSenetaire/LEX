from missingDataTrainingModule import PytorchDistributionUtils
from ..EvaluationUtils import define_target, calculate_cost
from ..utils.utils import *
from functools import partial, total_ordering
from .single_loss import SINGLE_LOSS

import numpy as np
import torch.nn.functional as F





class SEPARATE_LOSS(SINGLE_LOSS):
    def __init__(self, 
                interpretable_module,
                monte_carlo_gradient_estimator,
                baseline = None,
                fix_classifier_parameters = False,
                fix_selector_parameters = False,
                post_hoc = False,
                post_hoc_guidance = None,
                argmax_post_hoc = False,
                loss_function = None,
                loss_function_selection = None,
                nb_sample_z_monte_carlo = 1,
                nb_sample_z_iwae = 1,
                nb_sample_z_monte_carlo_classification = 1,
                nb_sample_z_iwae_classification = 1,
                ):

        super().__init__(interpretable_module=interpretable_module,
                        monte_carlo_gradient_estimator = monte_carlo_gradient_estimator,
                        baseline = baseline,
                        fix_classifier_parameters = fix_classifier_parameters,
                        fix_selector_parameters = fix_selector_parameters,
                        post_hoc = post_hoc,
                        post_hoc_guidance = post_hoc_guidance,
                        argmax_post_hoc = argmax_post_hoc,
                        loss_function = loss_function,
                        nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                        nb_sample_z_iwae = nb_sample_z_iwae,
                        )

        self.loss_function = loss_function
        if loss_function_selection is None :
            self.loss_function_selection = self.loss_function
        else :
            self.loss_function_selection = loss_function_selection

        assert self.interpretable_module.classification_distribution_module is not None, "classification_distribution_module must be defined"

        self.nb_sample_z_monte_carlo_classification = nb_sample_z_monte_carlo_classification
        self.nb_sample_z_iwae_classification = nb_sample_z_iwae_classification

    def _create_dic(self,
                loss_total,
                loss_rec_evalx,
                loss_rec,
                loss_reg,
                loss_selection,
                pi_list):
        dic = super()._create_dic(loss_total = loss_total, pi_list = pi_list, loss_rec = loss_rec, loss_reg = loss_reg, loss_selection = loss_selection)
        dic["loss_rec_evalx"] = loss_rec_evalx.detach().cpu().item()
        return dic


    def _train_step(self,
            data,
            target,
            dataset,
            index = None,
            need_dic = False,):

        self.interpretable_module.zero_grad()
        if self.monte_carlo_gradient_estimator.fix_n_mc :
            self.nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*self.nb_sample_z_iwae)
        
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc, dim_output= dataset.get_dim_output(),)
        
        data_expanded_classification, target_expanded_classification, index_expanded_classification, one_hot_target_expanded_classification = sampling_augmentation(data,
                                                                                                target = target,
                                                                                                index=index,
                                                                                                one_hot_target = one_hot_target,
                                                                                                mc_part = self.nb_sample_z_monte_carlo_classification,
                                                                                                iwae_part = self.nb_sample_z_iwae_classification,
                                                                                                )


        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                target = target,
                                                                                                index=index,
                                                                                                one_hot_target = one_hot_target,
                                                                                                mc_part = self.nb_sample_z_monte_carlo,
                                                                                                iwae_part = self.nb_sample_z_iwae,
                                                                                                )


        # Destructive module :
        log_pi_list, loss_reg = self.interpretable_module.selection_module(data)
        log_pi_list_classification = log_pi_list.unsqueeze(1).expand(batch_size, self.nb_sample_z_iwae_classification, -1)
        pi_list_classification = torch.exp(log_pi_list_classification)


        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, self.nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)


        #### TRAINING CLASSIFICATION :

        
        # Train classification module :
        p_z = self.interpretable_module.classification_distribution_module(pi_list_classification)
        z = self.interpretable_module.classification_distribution_module.sample(sample_shape = (self.nb_sample_z_monte_carlo_classification,))
        loss_classification = calculate_cost(mask_expanded = z,
                        interpretable_module = self.interpretable_module,
                        data_expanded = data_expanded_classification,
                        target_expanded = target_expanded_classification,
                        index_expanded = index_expanded_classification,
                        one_hot_target_expanded = one_hot_target_expanded_classification,
                        dim_output = dataset.get_dim_output(),
                        loss_function = self.loss_function,
                        )

        loss_classification = loss_classification.mean(axis = 0) # Mean on MC Samples here

        if not self.fix_classifier_parameters :
            torch.mean(loss_classification, axis=0).backward()
            self.optim_classification.step()
            self.interpretable_module.zero_grad()

        #### TRAINING SELECTION :

        cost_calculation = partial(calculate_cost,
                        interpretable_module = self.interpretable_module,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = self.loss_function_selection,
                        )

        loss_s, loss_f = self.monte_carlo_gradient_estimator(cost_calculation, pi_list, self.nb_sample_z_monte_carlo)


        loss_total = loss_s #  How to treat differently for REINFORCE or REPARAM ?

        if loss_reg is not None :
            loss_total += loss_reg

        if not self.fix_selector_parameters :
            torch.mean(loss_total).backward()
            self.optim_selection.step()
            if self.optim_distribution_module is not None :
                self.optim_distribution_module.step()

        if need_dic :
            dic = self._create_dic(loss_total = torch.mean(loss_s + loss_reg + loss_classification),
                                    loss_rec_evalx = torch.mean(loss_classification),
                                    loss_rec = torch.mean(loss_f),
                                    loss_reg = torch.mean(loss_reg),
                                    loss_selection = torch.mean(loss_s),
                                    pi_list = torch.exp(log_pi_list))
        else :
            dic = {}


        return dic


