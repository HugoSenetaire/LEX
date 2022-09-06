from missingDataTrainingModule import PytorchDistributionUtils
from ..utils.utils import *
from .fixed_selection_training import EVAL_X

import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class COUPLED_SELECTION(nn.Module):
    """ Abstract class to help the classification of the different module """

    def __init__(self,
                prediction_module,
                selection_module,
                distribution_module,
                reshape_mask_function = None,
                **kwargs):
        super(COUPLED_SELECTION, self).__init__()
        self.prediction_module = prediction_module
        self.selection_module = selection_module
        self.distribution_module = distribution_module
        self.reshape = reshape_mask_function
    
    def __call__(self, data, index = None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,):
        data_expanded = extend_input(data, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        index_expanded = extend_input(index, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        # Selection Module :
        batch_size = data.shape[0]
        log_pi_list, loss_reg = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)

        p_z = self.distribution_module(probs = pi_list)
        mask = p_z.sample((nb_sample_z_monte_carlo,))
        mask = self.reshape(mask)

        # Classification Module :
        log_y_hat, regul_classification = self.prediction_module(data_expanded, mask = mask, index = index_expanded)
        return log_y_hat, regul_classification, mask, loss_reg, p_z

    def sample_z(self, data, index, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1):
        batch_size = data.shape[0]
        log_pi_list, _ = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)
        p_z = self.distribution_module(probs = pi_list)
        z = self.distribution_module.sample((nb_sample_z_monte_carlo,))
        return z
        
        



class DECOUPLED_SELECTION(COUPLED_SELECTION):
    def __init__(self,
                prediction_module,
                selection_module,
                distribution_module,
                classification_distribution_module = PytorchDistributionUtils.wrappers.FixedBernoulli(),
                reshape_mask_function = None,
               ):

        super().__init__(prediction_module = prediction_module,
                        selection_module = selection_module,
                        distribution_module = distribution_module,
                        reshape_mask_function = reshape_mask_function,
                        )

        self.classification_distribution_module = classification_distribution_module
        self.EVALX = EVAL_X(prediction_module = prediction_module,
                    fixed_distribution = self.classification_distribution_module,
                    reshape_mask_function = self.reshape,
                    mask_dimension= self.selection_module.selector.output_size,)