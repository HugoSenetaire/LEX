import torch

from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented
from .gaussian_dataset import GaussianDataset


def getProbA(X):
    fa = torch.exp(X[:,0]*X[:,1])
    b_fa = 1/(1+fa)
    sel = torch.zeros_like(X)
    sel[:,:2] = 1
    return fa, b_fa, sel

def getProbB(X):
    fb = torch.exp(X[:,2:6].pow(2).sum(axis = 1))
    b_fb = 1/(1+fb)
    sel = torch.zeros_like(X)
    sel[:,2:6] = 1
    return fb, b_fb, sel 

def getProbC(X):
    fc = torch.exp(-10*torch.sin(0.2*X[:,6]) + torch.abs(X[:,7]) + X[:,8] + torch.exp(-X[:,9])-2.4)
    b_fc = 1/(1+fc)
    sel = torch.zeros_like(X)
    sel[:,6:10] = 1
    return fc, b_fc, sel

def generate_Y(X, nb_sample_train = 10000, nb_sample_test = 10000, getProb1 = getProbA, getProb2 = getProbB):
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

    X_train = X[:nb_sample_train,:]
    X_test = X[nb_sample_train:,:]
    Y_train = Y[:nb_sample_train,]
    Y_test = Y[nb_sample_train:,]

    sel_train = selection[:nb_sample_train,:]
    sel_test = selection[nb_sample_train:,:]
    
    return X_train, Y_train, sel_train, X_test, Y_test, sel_test

class S_1(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                give_index = False,
                noise_function = None,
                **kwargs):
        super().__init__(mean = mean, cov=cov, covariance_type = covariance_type, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)
        print(f"Given cov is {self.cov}")
        self.nb_dim = 11
        self.nb_classes = 2
        self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generate_Y(X = self.X,nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test, getProb1= getProbA, getProb2=getProbB)
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)


class S_2(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                give_index = False,
                noise_function = None,
                **kwargs):
        super().__init__(mean = mean, cov=cov, covariance_type = covariance_type, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)
        print(f"Given cov is {self.cov}")
        self.nb_dim = 11
        self.nb_classes = 2
        self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generate_Y(X = self.X,nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test, getProb1= getProbA, getProb2=getProbC)
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)

class S_3(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                give_index = False,
                noise_function = None,
                **kwargs):
        super().__init__(mean = mean, cov=cov, covariance_type = covariance_type, nb_sample_train = nb_sample_train, nb_sample_test = nb_sample_test, give_index = give_index, noise_function = noise_function, **kwargs)
        self.nb_dim = 11
        self.nb_classes = 2
        print(f"Given cov is {self.cov}")
        self.data_train, self.target_train, self.optimal_S_train, self.data_test, self.target_test, self.optimal_S_test = generate_Y(X = self.X,nb_sample_train = self.nb_sample_train, nb_sample_test = self.nb_sample_test, getProb1= getProbB, getProb2=getProbC)
        self.dataset_train = TensorDatasetAugmented(self.data_train, self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(self.data_test, self.target_test, give_index = self.give_index)



