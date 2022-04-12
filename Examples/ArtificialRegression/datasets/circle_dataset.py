import torch
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split


from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented

torch.pi = torch.tensor(3.1415)
np.random.seed(0)
torch.manual_seed(0)



class CircleDataset(ArtificialDataset):
    def __init__(self, nb_sample_train = 40000, nb_sample_test=10000, center = [0,0], factor =.6, noise = False, noise_function = None, **kwargs):
        super().__init__(nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = False, noise_function = noise_function, **kwargs)
        self.noise = noise
        self.factor = factor
        self.center = center

     
        total_samples = self.nb_sample_train + self.nb_sample_test
        test_size = self.nb_sample_test/float(total_samples)

        self.data, self.targets = datasets.make_circles(n_samples=total_samples, factor=.6,
                                      noise=noise)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(
            self.data, self.targets, test_size=test_size, random_state=0)

        self.data_train = torch.tensor(self.data_train)
        self.data_test = torch.tensor(self.data_test)
        self.targets_train = torch.tensor(self.targets_train)
        self.targets_test = torch.tensor(self.targets_test)

        self.dataset_train = TensorDatasetAugmented(self.data_train, self.targets_train, noise_function = noise_function)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.targets_test, noise_function= noise_function)


    def impute(self, value,  mask, index = None , dataset_type= None):
        mask_aux = mask.detach()
        value_aux = value.detach()
        ratio = torch.tensor(self.factor)
        batch_size = value.shape[0]

        x_1_missing = torch.where(mask_aux[:,0] == 0, True, False)
        x_2_missing = torch.where(mask_aux[:,1] == 0, True, False)

        both_missing = torch.where(x_1_missing, x_2_missing, False)
        none_missing = torch.where(x_1_missing, False , True)
        none_missing = torch.where(x_2_missing, False, none_missing).unsqueeze(-1).expand(-1,2)
        ratio_vector = torch.empty(0).new_full(value.shape, ratio).cuda()

        x_1_missing = torch.where(both_missing, False, x_1_missing)
        x_1_missing_sup_r1 = torch.where(torch.abs(value[:,1])>ratio_vector[:,1], x_1_missing, False)
        
        x_2_missing = torch.where(both_missing, False, x_2_missing)
        x_2_missing_sup_r2 = torch.where(torch.abs(value[:,0])>ratio_vector[:,1], x_2_missing, False)

        both_missing = both_missing.unsqueeze(-1).expand(-1,2)

        complete_output = value_aux.clone()

        # First both_missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size) * 0.5).unsqueeze(-1).expand(-1,2).cuda()
        uniform = 2*torch.pi* torch.rand(batch_size).cuda()
        new_output = (bernoulli + (1-bernoulli) * ratio) * torch.cat([torch.cos(uniform).unsqueeze(-1), torch.sin(uniform).unsqueeze(-1)],-1)
        complete_output = torch.where(both_missing, new_output, complete_output)

        # x_1 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_sup_r1 = (2*bernoulli-1) * torch.sqrt(1 - value[:,1]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_1_missing_new_value_inf_r1 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,1])**2,torch.zeros(batch_size).cuda()))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()

        complete_output[:,0] = torch.where(x_1_missing,
                            bernoulli * x_1_missing_new_value_sup_r1 + (1-bernoulli) * x_1_missing_new_value_inf_r1,
                            complete_output[:,0])

        complete_output[:,0] = torch.where(x_1_missing_sup_r1, x_1_missing_new_value_sup_r1,
                            complete_output[:,0]) 


        # x_2 missing :
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_2_missing_new_value_sup_r2 = (2*bernoulli-1) * torch.sqrt(1 - value[:,0]**2) 
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()
        x_2_missing_new_value_inf_r2 = (2*bernoulli-1) * torch.sqrt(torch.maximum(ratio**2 - (value[:,0])**2,torch.zeros(batch_size).cuda()))
        bernoulli = torch.bernoulli(torch.ones(batch_size)*0.5).cuda()

        complete_output[:,1] = torch.where(x_2_missing,
                            bernoulli * x_2_missing_new_value_sup_r2 + (1-bernoulli) * x_2_missing_new_value_inf_r2,
                            complete_output[:,1])

        complete_output[:,1] = torch.where(x_2_missing_sup_r2, x_2_missing_new_value_sup_r2,
                            complete_output[:,1])


        # None missing 
        complete_output = torch.where(none_missing, value, complete_output)


        return complete_output

