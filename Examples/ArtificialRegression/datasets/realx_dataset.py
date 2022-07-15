import torch

from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented
from .gaussian_dataset import GaussianDataset
from .utils import getProbA, getProbB, getProbC, f_a, f_b, f_c



def generate_Y_classification(X, nb_sample_train = 10000, nb_sample_test = 10000, getProb1 = getProbA, getProb2 = getProbB):
    assert(len(X) == nb_sample_test + nb_sample_train)
    _, prob1, sel1 = getProb1(X)
    _, prob2, sel2 = getProb2(X)
    aux = X[:,10]<0
    prob_total = torch.where(aux, prob1, prob2)
    
    # aux = torch.reshape(aux, (aux.shape[0],1)).repeat(repeats = X.shape[1], axis = 1)
    aux = aux.unsqueeze(1).expand(X.shape[0], X.shape[1])
    selection = torch.where(aux, sel1, sel2)
    selection[:,10] = 1
    Y = torch.rand(size = prob_total.shape)
    Y = torch.where(Y<prob_total, torch.ones_like(prob_total, dtype = torch.int64), torch.zeros_like(prob_total, dtype = torch.int64))

    data_train = X[:nb_sample_train,:]
    data_test = X[nb_sample_train:,:]
    target_train = Y[:nb_sample_train,]
    target_test = Y[nb_sample_train:,]

    sel_train = selection[:nb_sample_train,:]
    sel_test = selection[nb_sample_train:,:]
    
    return data_train, target_train, sel_train, data_test, target_test, sel_test

def generate_Y_regression(X, epsilon_sigma = 1.0, nb_sample_train = 10000, nb_sample_test = 10000, getProb1 = getProbA, getProb2 = getProbB, scaling_regression = True):
    assert(len(X) == nb_sample_test + nb_sample_train)
    f1, prob1, sel1 = getProb1(X)
    f2, prob2, sel2 = getProb2(X)
    aux = X[:,10]<0

    if scaling_regression :
        f_total = torch.where(aux, prob1, prob2)
    else :
        f_total = torch.where(aux, f1, f2)
    
    aux = aux.unsqueeze(1).expand(X.shape[0], X.shape[1])
    selection = torch.where(aux, sel1, sel2)
    selection[:,10] = 1

    
    Y = torch.distributions.normal.Normal(f_total, epsilon_sigma).sample()
    data_train = X[:nb_sample_train,:]
    data_test = X[nb_sample_train:,:]
    target_train = Y[:nb_sample_train,]
    target_test = Y[nb_sample_train:,]

    sel_train = selection[:nb_sample_train,:]
    sel_test = selection[nb_sample_train:,:]
    
    return data_train, target_train, sel_train, data_test, target_test, sel_test

class S_init(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                classification = True,
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                **kwargs):
        super().__init__(mean = mean, cov=cov, covariance_type = covariance_type, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)
        print(f"Given cov is {self.cov}")

        self.dim_input = 11


        self.classification = classification
        self.epsilon_sigma = epsilon_sigma
        self.scaling_regression = scaling_regression
        if self.classification :
            self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generate_Y_classification(X = self.X,
                                                                                                                            nb_sample_train = self.nb_sample_train,
                                                                                                                            nb_sample_test = self.nb_sample_test,
                                                                                                                            getProb1= self.getProb1,
                                                                                                                            getProb2=self.getProb2)
            self.nb_classes = 2
        else :
            self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generate_Y_regression(X = self.X,
                                                                                                                            epsilon_sigma = self.epsilon_sigma,
                                                                                                                            nb_sample_train = self.nb_sample_train,
                                                                                                                            nb_sample_test = self.nb_sample_test,
                                                                                                                            getProb1 = self.getProb1,
                                                                                                                            getProb2 = self.getProb2,
                                                                                                                            scaling_regression = self.scaling_regression)
            self.nb_classes = 1
        self.dataset_train = TensorDatasetAugmented(x = self.data_train, y = self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(x = self.data_test, y = self.target_test, give_index = self.give_index)


class S_1(S_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                classification = True,
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                **kwargs):
        self.getProb1 = getProbA
        self.getProb2 = getProbB
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                classification = classification,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                **kwargs)

  
class S_2(S_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                classification = True,
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                **kwargs):
        self.getProb1 = getProbA
        self.getProb2 = getProbC
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                classification = classification,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                **kwargs)


class S_3(S_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                classification = True,
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                **kwargs):
        self.getProb1 = getProbB
        self.getProb2 = getProbC
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                classification = classification,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                **kwargs)
  
