from functools import partial, total_ordering
from ..utils import *
from ..EvaluationUtils import *


import numpy as np
import torch.nn.functional as F



class SINGLE_LOSS():
    """ Abstract class to help the classification of the different module """

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
                nb_sample_z_monte_carlo = 1,
                nb_sample_z_iwae = 1,):

        self.interpretable_module = interpretable_module
        self.monte_carlo_gradient_estimator = monte_carlo_gradient_estimator(distribution = self.interpretable_module.distribution_module)

        self.use_cuda = False
        self.compiled = False
        self.baseline = baseline

        self.loss_function = loss_function
        self.nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.nb_sample_z_iwae = nb_sample_z_iwae

        self.fix_classifier_parameters = fix_classifier_parameters
        self.fix_selector_parameters = fix_selector_parameters
        self.post_hoc_guidance = post_hoc_guidance
        self.post_hoc = post_hoc
        self.argmax_post_hoc = argmax_post_hoc
        
        if self.post_hoc_guidance is not None :
            for param in self.post_hoc_guidance.parameters():
                param.requires_grad = False

        if self.post_hoc and(self.post_hoc_guidance is None) and self.fix_classifier_parameters :
            self.post_hoc_guidance = self.interpretable_module.prediction_module

        if self.post_hoc and (self.post_hoc_guidance is None) and (not self.fix_classifier_parameters):
            raise AttributeError("You can't have post-hoc without a post hoc guidance or fixing the classifier parameters")

    def cuda(self):
        if torch.cuda.is_available():
            self.use_cuda = True
            self.interpretable_module.cuda()

    def define_input(self, data, target, index, dataset, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,):
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc, dim_output= dataset.get_dim_output(),)

        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                        target = target,
                                                                                                        index=index,
                                                                                                        one_hot_target = one_hot_target,
                                                                                                        mc_part = nb_sample_z_monte_carlo,
                                                                                                        iwae_part= nb_sample_z_iwae,
                                                                                                        )

        return data, target, index, one_hot_target, data_expanded, target_expanded, index_expanded, one_hot_target_expanded, batch_size

    def compile(self, optim_classification, optim_selection, scheduler_classification = None, scheduler_selection = None, optim_baseline = None, scheduler_baseline = None, optim_distribution_module = None, scheduler_distribution_module = None, **kwargs):
        self.optim_classification = optim_classification
        if self.optim_classification is None :
            self.fix_classifier_parameters = True
        self.scheduler_classification = scheduler_classification
        self.optim_selection = optim_selection
        if self.optim_selection is None :
            self.fix_selector_parameters = True
        self.scheduler_selection = scheduler_selection
        self.optim_distribution_module = optim_distribution_module
        self.scheduler_distribution_module = scheduler_distribution_module

        self.compiled = True


    def scheduler_step(self):
        assert(self.compiled)
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        if self.scheduler_selection is not None :
            self.scheduler_selection.step()
        if self.scheduler_distribution_module is not None :
            self.scheduler_distribution_module.step()
        self.interpretable_module.distribution_module.update_distribution()

    def optim_step(self):
        assert(self.compiled)
        if (not self.fix_classifier_parameters) and (self.optim_classification is not None) :
            self.optim_classification.step()
        if (not self.fix_selector_parameters) and (self.optim_selection is not None) :
            self.optim_selection.step()
        if self.optim_distribution_module is not None :
            self.optim_distribution_module.step()

    def _create_dic(self, loss_total, pi_list, loss_rec = None, loss_reg = None, loss_selection = None, ):
        dic = {}
        dic["loss_total"] = loss_total.detach().cpu().item()
        dic["mean_pi_list"] = torch.mean(torch.mean(pi_list.flatten(1),1)).item()
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list.flatten(1),quantiles,dim=1,keepdim = True)
        dic["pi_list_median"] = torch.mean(q[1]).item()
        dic["pi_list_q1"] = torch.mean(q[0]).item()
        dic["pi_list_q2"] = torch.mean(q[2]).item()
        if self.interpretable_module.prediction_module.imputation.has_constant():
            if torch.is_tensor(self.interpretable_module.prediction_module.imputation.get_constant()):
                dic["constantLeanarble"]= self.interpretable_module.prediction_module.imputation.get_constant().item()
        
        dic["loss_rec"] = get_item(loss_rec)
        dic["loss_reg"] = get_item(loss_reg)
        dic["loss_selection"] = get_item(loss_selection)
        return dic

    
    def _train_step(self, data, target, dataset, index = None,  need_dic = False):
        self.interpretable_module.zero_grad()
        if self.monte_carlo_gradient_estimator.fix_n_mc : # If we fix number of samples to all samples
            self.nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*self.nb_sample_z_iwae)

        data, target, index, one_hot_target, data_expanded, target_expanded, index_expanded, one_hot_target_expanded, batch_size = self.define_input(data,
                                                                                                                                                target,
                                                                                                                                                index,
                                                                                                                                                dataset,
                                                                                                                                                nb_sample_z_monte_carlo = self.nb_sample_z_monte_carlo,
                                                                                                                                                nb_sample_z_iwae = self.nb_sample_z_iwae,
                                                                                                                                            )
        
        # Selection Module :
        log_pi_list, loss_reg = self.interpretable_module.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, self.nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)
        cost_calculation = partial(calculate_cost,
                        interpretable_module = self.interpretable_module,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = self.loss_function,
                        )
        loss_s, loss_f = self.monte_carlo_gradient_estimator(cost_calculation, pi_list, self.nb_sample_z_monte_carlo)
        
        if self.monte_carlo_gradient_estimator.combined_grad_f_s : # Different if we have pathwise gradient or not, at some point, we might need a loss in f (prediction) and a loss in s (selection)
            loss_total = loss_reg + loss_s # How to treat differently for REINFORCE or REPARAM ?
        else :
            loss_total = loss_reg + loss_s + loss_f

        torch.mean(loss_total).backward()
        self.optim_step()

        if need_dic :
            dic = self._create_dic(loss_total = torch.mean(loss_total),
                        loss_rec = torch.mean(loss_f),
                        loss_reg = torch.mean(loss_reg),
                        loss_selection = torch.mean(loss_s),
                        pi_list = torch.exp(log_pi_list),
                        )
        else :
            dic = {}
        return dic

    