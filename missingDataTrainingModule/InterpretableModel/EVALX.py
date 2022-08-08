from ..utils import define_target, continuous_NLLLoss, MSELossLastDim, NLLLossAugmented, AccuracyLoss, calculate_cost, multiple_test, test_train_loss
from ..utils.utils import *
from ..PytorchDistributionUtils import *
import torch.nn.functional as F
import torch.nn as nn




class ordinaryTraining(nn.Module):
    def __init__(self, classification_module,): 
        super(ordinaryTraining, self).__init__()
        self.classification_module = classification_module
        


    def __call__(self, data, index = None,):
        log_y_hat, regul_classification = self.classification_module(data, index= index)
        return log_y_hat, regul_classification, None, None, None





class trueSelectionTraining(ordinaryTraining):
    def __init__(self, classification_module, dataset):   
        super().__init__(classification_module, )
        self.dataset = dataset

    def reshape(self, mask):
        return mask
    
    def __call__(self, data, index=None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1):
        true_mask = self.dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        data_expanded = extend_input(data, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        true_mask_expanded = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        index_expanded = extend_input(index, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        log_y_hat, regul_classification = self.classification_module(data_expanded, mask = true_mask_expanded, index=index_expanded)
        return log_y_hat, regul_classification, true_mask_expanded, None, None


    def sample_z(self, data, index, nb_sample_z_monte_carlo =1, nb_sample_z_iwae =1 ):
        true_mask = self.dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        return true_mask
    




class EVAL_X(ordinaryTraining):
    def __init__(self,
                    classification_module,
                    fixed_distribution = wrappers.FixedBernoulli(),
                    reshape_mask_function = None, 
                    ):
        super().__init__(classification_module, )
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
        batch_size = data.shape[0]

        nb_sample_z_monte_carlo_classification, nb_sample_z_iwae_classification = nb_sample_z_monte_carlo*nb_sample_z_iwae, 1
        
        # Destructive module
        p_z = self.fixed_distribution(torch.zeros(batch_size, nb_sample_z_iwae_classification, *self.classification_module.classifier.input_size[1:]).to(data.device))
        # Train classification module :
        mask = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo_classification,))
        mask = self.reshape(mask)

        log_y_hat, regul_classification = self.classification_module(data_expanded, mask = mask, index = index_expanded)
        return log_y_hat, regul_classification, mask, None, p_z


    def sample_z(self, data, target, index, dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae):
        batch_size = data.shape[0]
        p_z = self.fixed_distribution(torch.zeros(batch_size, nb_sample_z_iwae, 1, *self.classification_module.classifier.input_size[1:]).to(data.device))
        z = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo,))
        return z
    
