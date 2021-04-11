import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

from .classification_network import *
from .imputation import *



class ClassificationModule():

    def __init__(self, classifier, imputation = None, imputation_reg = None, feature_extractor = None):
        self.classifier = classifier
        self.classifier_fixed = False
        self.imputation = imputation
        if self.imputation is None :
            self.need_imputation = False
        else :
            self.need_imputation = True

        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_extraction = False
        
        else :
            self.need_extraction = True

    
        self.kernel_updated = False

    def kernel_update(self, kernel_patch, stride_patch):
        if self.need_imputation:
            self.imputation.kernel_update(kernel_patch, stride_patch)
    

    def train(self):
        if self.need_imputation :
            self.imputation.train()
        self.classifier.train()

    def eval(self):
        if self.need_imputation :
            self.imputation.eval()
        self.classifier.eval()

    def cuda(self):
        self.classifier = self.classifier.cuda()
        if self.need_imputation  :
            self.imputation.cuda()

    def zero_grad(self):
        self.classifier.zero_grad()
        if self.need_imputation  :
            self.imputation.zero_grad()

    def parameters(self):
        # if self.need_imputation and self. 
        # TODO Check learnable parameters of imputation method
        
        if self.need_imputation and self.imputation.is_learnable() :
            list_param = [{"params" : self.classifier.parameters()}]
            for element in self.imputation.get_learnable_parameter():
                list_param.append(element)
            return list_param
        else :
            return self.classifier.parameters()

    def fix_parameters(self):
        self.classifier_fixed = True
        for param in self.classifier.parameters():
         param.requires_grad = False


    def __call__(self, data, sample_b = None):

        if sample_b is not None :
            sample_b = sample_b.reshape(data.shape)



        if self.imputation is not None and sample_b is None :
            raise AssertionError("If using imputation, you should give a sample of bernoulli or relaxed bernoulli")
        elif self.imputation is not None and sample_b is not None :
            x_imputed, loss_reconstruction = self.imputation.impute(data, sample_b)
            if self.need_extraction :
                x_imputed = self.feature_extractor(x_imputed)
            y_hat = self.classifier(x_imputed)

            
        else :
            if self.need_extraction :
                data = self.feature_extractor(data)
            y_hat = self.classifier(data)
            loss_reconstruction = torch.zeros((1)).cuda()
        return y_hat, loss_reconstruction


