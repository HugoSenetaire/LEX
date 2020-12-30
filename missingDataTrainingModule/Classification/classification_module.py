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

    def train(self):
        if self.need_imputation and self.imputation.is_learnable() :
            self.imputation.get_learnable_parameter().requires_grad_(True)
        self.classifier.train()

    def eval(self):
        if self.need_imputation and self.imputation.is_learnable() :
            self.imputation.get_learnable_parameter().requires_grad_(False)
        self.classifier.eval()

    def cuda(self):
        self.classifier.cuda()
        if self.need_imputation and self.imputation.is_learnable() :
            self.imputation.get_learnable_parameter().cuda()

    def zero_grad(self):
        self.classifier.zero_grad()
        if self.need_imputation and self.imputation.is_learnable() :

            self.imputation.zero_grad()

    def parameters(self):
        # if self.need_imputation and self. 
        # TODO Check learnable parameters of imputation method
        

        if self.need_imputation and self.imputation.is_learnable() :
            return [
                {"params" : self.classifier.parameters()},
                {"params" : self.imputation.get_learnable_parameter()},
            ]
        else :
            return self.classifier.parameters()



    def __call__(self, data, sample_b = None):
        if self.imputation is not None and sample_b is None :
            raise AssertionError("If using imputation, you should give a sample of bernoulli or relaxed bernoulli")
        elif self.imputation is not None and sample_b is not None :
            x_imputed = self.imputation.impute(data, sample_b)
            y_hat = self.classifier(x_imputed)
        else :
            y_hat = self.classifier(data)
        return y_hat


