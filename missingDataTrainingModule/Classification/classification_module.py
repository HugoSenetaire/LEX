import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *

from .classification_network import *
from .imputation import *

import torch.nn as nn



class ClassificationModule(nn.Module):

    def __init__(self, classifier, imputation = None, imputation_reg = None, **kwargs):
        super(ClassificationModule, self).__init__()
        self.classifier = classifier
        self.classifier_fixed = False
        self.imputation = imputation
        if self.imputation is None :
            self.need_imputation = False
        else :
            self.need_imputation = True


        if imputation is not None :
            self.input_size = self.imputation.input_size
        else :
            self.input_size = self.classifier.input_size
        self.kernel_updated = False

    def get_imputation(self, data, mask, index = None):
        data = data.reshape(mask.shape) # Quick fix when the reshape function do not match the shape of the data (change the dataset might be better), TODO
        x_imputed, _ = self.imputation(data, mask, index)
        return x_imputed

    def __call__(self, data, mask = None, index = None):
        """ Using the data and the mask, do the imputation and classification 
        
        Parameters:
        -----------
        data : torch.Tensor of shape (Nexpectation*batch_size, channels, size_lists...)
            The data to be classified
        mask : torch.Tensor of shape (Nexpectation*batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (Nexpectation*batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        y_hat : torch.Tensor of shape (nb_imputation*Nexpectation*batch_size, nb_category)
            The output of the classification
        loss_reconstruction : torch.Tensor of shape (1)
            Some regularization term that can be added to the loss (for instance in the case of version Autoencoder regularisation)

        """
        
        if mask is not None :
            data = data.reshape(mask.shape) # Quick fix when the reshape function do not match the shape of the data (change the dataset might be better), TODO
        if self.imputation is not None and mask is not None :
            x_imputed, loss_reconstruction = self.imputation(data, mask, index)
            y_hat = self.classifier(x_imputed)
        else :
            y_hat = self.classifier(data)
            loss_reconstruction = torch.zeros((1))
            if y_hat.is_cuda :
                loss_reconstruction = loss_reconstruction.cuda()

        return y_hat, loss_reconstruction


