import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

from .classification_network import *
from .imputation import *



class ClassificationModule():

    def __init__(self, classifier, imputation = None):
        self.classifier = classifier
        self.imputation = imputation
        if self.imputation is None :
            self.need_imputation = False
        else :
            self.need_imputation = True
        self.kernel_updated = False

    def kernel_update(self, kernel_patch, stride_patch):
        if self.need_imputation:
            self.imputation.kernel_update(kernel_patch, stride_patch)
    

    def train(self):
        if self.need_imputation and self.imputation.is_learnable() :
            if self.imputation.has_constant():
                self.imputation.get_learnable_parameter().requires_grad_(True)
            else :
                self.imputation.train()
        self.classifier.train()

    def eval(self):
        if self.need_imputation and self.imputation.is_learnable() :
            if self.imputation.has_constant():
                self.imputation.get_learnable_parameter().requires_grad_(True)
            else :
                self.imputation.eval()
        self.classifier.eval()

    def cuda(self):
        self.classifier = self.classifier.cuda()
      
        if self.need_imputation and self.imputation.is_learnable() :
            self.imputation.cuda()

    def zero_grad(self):
        self.classifier.zero_grad()
        if self.need_imputation and self.imputation.is_learnable() :
            self.imputation.zero_grad()

    def parameters(self):
        # if self.need_imputation and self. 
        # TODO Check learnable parameters of imputation method
        

        if self.need_imputation and self.imputation.is_learnable() :
            if self.imputation.has_constant():
                return [
                    {"params" : self.classifier.parameters()},
                    {"params" : self.imputation.get_learnable_parameter()},
                ]
            else :
                list_param = [{"params" : self.classifier.parameters()}]
                for element in self.imputation.get_learnable_parameter():
                    list_param.append(element)
                return list_param
        else :
            return self.classifier.parameters()



    def __call__(self, data, sample_b = None):
        if len(sample_b.shape)>2:
            sample_b = sample_b.flatten(0,1)
        if self.imputation is not None and sample_b is None :
            raise AssertionError("If using imputation, you should give a sample of bernoulli or relaxed bernoulli")
        elif self.imputation is not None and sample_b is not None :
            x_imputed = self.imputation.impute(data, sample_b)
            y_hat = self.classifier(x_imputed)
        else :
            y_hat = self.classifier(data)
        return y_hat


