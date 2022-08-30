from ..utils.utils import *
from ..PytorchDistributionUtils import *
import torch.nn.functional as F
import torch.nn as nn




class PredictionCompleteModel(nn.Module):
    def __init__(self, prediction_module,): 
        super(PredictionCompleteModel, self).__init__()
        self.prediction_module = prediction_module
        


    def __call__(self, data, index = None,):
        log_y_hat, regul_classification = self.prediction_module(data, index= index)
        return log_y_hat, regul_classification, None, None, None





class trueSelectionCompleteModel(PredictionCompleteModel):
    def __init__(self, prediction_module, dataset):   
        super().__init__(prediction_module, )
        self.dataset = dataset

    def reshape(self, mask):
        return mask
    
    def __call__(self, data, index=None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1):
        true_mask = self.dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        data_expanded = extend_input(data, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        true_mask_expanded = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        index_expanded = extend_input(index, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        log_y_hat, regul_classification = self.prediction_module(data_expanded, mask = true_mask_expanded, index=index_expanded)
        return log_y_hat, regul_classification, true_mask_expanded, None, None


    def sample_z(self, data, index, nb_sample_z_monte_carlo =1, nb_sample_z_iwae =1 ):
        true_mask = self.dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        return true_mask
    
    


class EVAL_X(PredictionCompleteModel):
    def __init__(self,
                    prediction_module,
                    fixed_distribution = wrappers.FixedBernoulli(),
                    reshape_mask_function = None, 
                    ):
        super().__init__(prediction_module, )
        self.fixed_distribution = fixed_distribution
        self.reshape_mask_function = reshape_mask_function



    def reshape(self, z):
        if self.reshape_mask_function is not None:
            return self.reshape_mask_function(z)
        else :
            return z

    def __call__(self, data, index = None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1, ):   
        data_expanded = extend_input(data, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        index_expanded = extend_input(index, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        if index_expanded is not None :
            index_expanded = index_expanded.flatten(0,2)
        batch_size = data.shape[0]
        
        # Destructive module
        p_z = self.fixed_distribution(torch.zeros(batch_size, nb_sample_z_iwae, *self.prediction_module.classifier.input_size[1:]).to(data.device))
        # Train classification module :
        mask = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo,))
        mask = self.reshape(mask)

        
        
        log_y_hat, regul_classification = self.prediction_module(data_expanded.flatten(0,2), mask = mask, index = index_expanded)
        return log_y_hat, regul_classification, mask, None, p_z


    def sample_z(self, data, index, nb_sample_z_monte_carlo = 1 , nb_sample_z_iwae = 1):
        batch_size = data.shape[0]
        p_z = self.fixed_distribution(torch.zeros(batch_size, nb_sample_z_iwae, 1, *self.prediction_module.classifier.input_size[1:]).to(data.device))
        z = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo,))
        return z
    