import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from utils_missing import *

from .classification_network import *
from .imputation import *



class ClassificationModule():

    def __init__(self, classifier, imputation = None, imputation_reg = None, feature_extractor = None, use_cuda = False):
        self.classifier = classifier
        self.classifier_fixed = False
        self.imputation = imputation
        self.use_cuda = use_cuda
        if self.imputation is None :
            self.need_imputation = False
        else :
            self.need_imputation = True

        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_extraction = False
        
        else :
            self.need_extraction = True
        if imputation is not None :
            self.input_size = self.imputation.input_size
        else :
            self.input_size = self.classifier.input_size
        self.kernel_updated = False

    def kernel_update(self, kernel_patch, stride_patch):
        self.kernel_updated = True
        self.image = False
        if self.input_size is int or len(self.input_size)<=1:
            self.kernel_patch = 1
            self.stride_patch = 1
            try :
                self.nb_patch_x, self.nb_patch_y = int(self.input_size), None
            except :
                self.nb_patch_x, self.nb_patch_y = int(self.input_size[1]), None
        elif len(self.input_size)==2: # For protein like example (1D CNN) #TODO: really implement that ?
            self.kernel_patch = kernel_patch
            self.stride_patch = stride_patch
            self.nb_patch_x, self.nb_patch_y = int(self.input_size[1]), None 
        else :
            self.image = True
            assert(kernel_patch[0]>= stride_patch[0])
            assert(kernel_patch[1]>= stride_patch[1])
            assert(stride_patch[0]>0)
            assert(stride_patch[1]>0)
            self.kernel_patch = kernel_patch
            self.stride_patch = stride_patch
            self.nb_patch_x, self.nb_patch_y = calculate_pi_dimension(self.input_size, self.stride_patch)

    
  
    def patch_creation(self,sample_b):
        """ Recreating the heat-map of selection being given the original selection mask
        TODO : This is very slow, check how a convolution is done in Pytorch. Might need to use some tricks to accelerate here """
        assert(self.kernel_updated)
    
        if self.image :
            if self.kernel_patch == (1,1):
                return sample_b
            else :
                aux_sample_b = sample_b.reshape(-1, self.input_size[0], self.nb_patch_x*self.nb_patch_y)
                aux_sample_b = aux_sample_b.unsqueeze(2).expand(-1, -1, np.prod(self.kernel_patch), -1).flatten(1,2)
                new_sample_b = torch.nn.Fold((self.input_size[1], self.input_size[2]),self.kernel_patch, stride = self.stride_patch)(aux_sample_b)

            new_sample_b = new_sample_b.clamp(0.0, 1.0)    
        else :
            new_sample_b = sample_b
        return new_sample_b

    
    def readable_sample(self, sample_b):
        """ Use during test or analysis to make sure we have a mask dimension similar to the input dimension """
        return self.patch_creation(sample_b)         
            
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

    def multiple_channel(self, data, sample_b):
        if len(data.shape)>2 and data.shape[1]>1 : # If multiple channels
            wanted_transform = tuple(np.insert(-np.ones(len(sample_b.shape),dtype = int),1,data.shape[1]))
            sample_b = sample_b.unsqueeze(1).expand(wanted_transform)
            sample_b = sample_b.flatten(1,2)
        return sample_b

    def prepare_mask(self, data, sample_b):
        sample_b = self.multiple_channel(data, sample_b)
        sample_b = self.patch_creation(sample_b)
        sample_b = sample_b.reshape(data.shape)
        return sample_b


    def __call__(self, data, sample_b = None, index = None):
        if sample_b is not None :
            sample_b = self.prepare_mask(data, sample_b)
        # if self.imputation is not None and sample_b is None :
            # print("If using imputation, you should give a sample of bernoulli or relaxed bernoulli")
            # raise AssertionError("If using imputation, you should give a sample of bernoulli or relaxed bernoulli")
        
    
        if self.imputation is not None and sample_b is not None :
            x_imputed, loss_reconstruction = self.imputation.impute(data, sample_b, index)
            # x_imputed_aux = x_imputed.cpu().detach().numpy()
            # fig, (ax1, ax2, ax3)= plt.subplots(1,3)
            # ax1.imshow(data[0].cpu().detach().numpy().reshape((28,28)), cmap='gray')
            # ax2.imshow(sample_b[0].cpu().detach().numpy().reshape((28,28)), cmap='gray')
            # ax3.imshow(x_imputed_aux[0].reshape((28,28)), cmap='gray')
            # plt.show()
            if self.need_extraction :
                x_imputed = self.feature_extractor(x_imputed)
            y_hat = self.classifier(x_imputed)

            
        else :
            if self.need_extraction :
                data = self.feature_extractor(data)
            y_hat = self.classifier(data)
            loss_reconstruction = torch.zeros((1))
            if self.use_cuda :
                loss_reconstruction = loss_reconstruction.cuda()
        return y_hat, loss_reconstruction


