import torch

from .artificial_dataset import ArtificialDataset
from .tensor_dataset_augmented import TensorDatasetAugmented
from .gaussian_dataset import GaussianDataset
from .utils import getProbA, getProbB, getProbC, f_a, f_b, f_c

"""
Generating Synthetic Data for Synthetic Examples
There are 6 Synthetic Datasets

X ~ N(0,I) where d = 100

Y = 1/(1+logit)
- Syn1: logit = exp(X1 * X2)
- Syn2: logit = exp(X3^2 + X4^2 + X5^2 + X6^2 -4)
- Syn3: logit = -10 sin(2 * X7) + 2|X8| + X9 + exp(-X10) - 2.4
- Syn4: If X11 < 0, Syn1, X11 >= Syn2
- Syn4: If X11 < 0, Syn1, X11 >= Syn3
- Syn4: If X11 < 0, Syn2, X11 >= Syn3
""" 
#%% Necessary packages
import numpy as np 

#%% X Generation
def generate_X (n=10000):
    
    X = np.random.randn(n, 11)
    
    return X

#%% Basic Label Generation (Syn1, Syn2, Syn3)
'''
X: Features
data_type: Syn1, Syn2, Syn3
'''
def Basic_Label_Generation(X, data_type):
    print("DATA TYPE", data_type)

    # number of samples
    n = len(X[:,0])
    
    # Logit computation
    # 1. Syn1
    if (data_type == 'Syn1'):
        logit = np.exp(X[:,0]*X[:,1])
        
    # 2. Syn2
    elif (data_type == 'Syn2'):       
        logit = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
        
    # 3. Syn3
    elif (data_type == 'Syn3'):
        logit = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 
        
    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape( (1 / (1+logit)), [n,1])
    prob_0 = np.reshape( (logit / (1+logit)), [n,1])
    
    # Probability output
    prob_y = np.concatenate((prob_0,prob_1), axis = 1)
    
    # Sampling from the probability
    y = np.zeros([n,2])
    y[:,0] = np.reshape(np.random.binomial(1, prob_0), [n,])
    y[:,1] = 1-y[:,0]

    return y, prob_y
    
#%% Complex Label Generation (Syn4, Syn5, Syn6)

def Complex_Label_Generation(X, data_type):
    
    # number of samples
    n = len(X[:,0])
    
    # Logit generation
    # 1. Syn4
    if (data_type == 'Syn4'):
        logit1 = np.exp(X[:,0]*X[:,1])
        logit2 = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0)
        print("LOGIT1: ", logit1)
        print("LOGIT2: ", logit2)
    
    # 2. Syn5
    elif (data_type == 'Syn5'):
        logit1 = np.exp(X[:,0]*X[:,1])
        logit2 = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 
    
    # 3. Syn6
    elif (data_type == 'Syn6'):
        logit1 = np.exp(np.sum(X[:,2:6]**2, axis = 1) - 4.0) 
        logit2 = np.exp(-10 * np.sin(0.2*X[:,6]) + abs(X[:,7]) + X[:,8] + np.exp(-X[:,9])  - 2.4) 

    # Based on X[:,10], combine two logits        
    idx1 = (X[:,10]< 0)*1
    idx2 = (X[:,10]>=0)*1
    
    logit = logit1 * idx1 + logit2 * idx2
        
    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape( (1 / (1+logit)), [n,1])
    prob_0 = np.reshape( (logit / (1+logit)), [n,1])
    
    # Probability output
    prob_y = np.concatenate((prob_0,prob_1), axis = 1)
    
    # Sampling from the probability
    y = np.zeros([n,2])
    y[:,0] = np.reshape(np.random.binomial(1, prob_0), [n,])
    y[:,1] = 1-y[:,0]

    return y, prob_y

#%% Ground truth Variable Importance

def Ground_Truth_Generation(X, data_type):

    # Number of samples and features
    n = len(X[:,0])
    d = len(X[0,:])

    # Output initialization
    out = np.zeros([n,d])
    
    # Index
    if (data_type in ['Syn4','Syn5','Syn6']):        
        idx1 = np.where(X[:,10]< 0)[0]
        idx2 = np.where(X[:,10]>=0)[0]
        out[:,10] = 1
    
    # For each data_type
    # Simple
    if (data_type == 'Syn1'):
        out[:,:2] = 1
    elif (data_type == 'Syn2'):
        out[:,2:6] = 1
    elif (data_type == 'Syn3'):
        out[:,6:10] = 1
        
    # Complex
    else :
        out[:, 10] = 1
        if (data_type == 'Syn4'):        
            out[idx1,:2] = 1
            out[idx2,2:6] = 1
        elif (data_type == 'Syn5'):        
            out[idx1,:2] = 1
            out[idx2,6:10] = 1
        elif (data_type == 'Syn6'):        
            out[idx1,2:6] = 1
            out[idx2,6:10] = 1
    return out

    
#%% Generate X and Y
'''
n: Number of samples
data_type: Syn1 to Syn6
out: Y or Prob_Y
'''    
    
def generate_data(n=10000, data_type='Syn4', seed = 0, out = 'Y'):

    # For same seed
    np.random.seed(seed)

    # X generation
    X = generate_X(n)

    # Y generation
    if (data_type in ['Syn1','Syn2','Syn3']):
        Y, Prob_Y = Basic_Label_Generation(X, data_type)
        
    elif (data_type in ['Syn4','Syn5','Syn6']):
        Y, Prob_Y = Complex_Label_Generation(X, data_type)
    
    # Output
    if out == 'Prob':
        Y_Out = Prob_Y
    elif out == 'Y':
        Y_Out = Y
        
    # Ground truth
    print("CREATION", Y_Out.mean(axis=0))
    Ground_Truth = Ground_Truth_Generation(X, data_type)
        
    return X, Y_Out, Ground_Truth
    

class Syn_init(GaussianDataset):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                data_type = None,
                noise_function = None,
                train_seed = 0,
                test_seed = 1,
                **kwargs):
        super().__init__(mean = mean,
                        cov=cov,
                        covariance_type = covariance_type,
                        nb_sample_train = nb_sample_train,
                        nb_sample_test = nb_sample_test,
                        give_index = give_index,
                        noise_function = noise_function,
                        train_seed=train_seed,
                        test_seed=test_seed,
                        **kwargs)

        print(f"Given cov is {self.cov}")

        self.dim_input = 11
        self.epsilon_sigma = epsilon_sigma
        self.scaling_regression = scaling_regression
        self.data_type = data_type
        
        self.nb_classes = 2
        self.data_train, self.target_train, self.optimal_S_train = generate_data(n = self.nb_sample_train, data_type = self.data_type,seed = train_seed,)
        self.data_test, self.target_test, self.optimal_S_test = generate_data(n = self.nb_sample_test, data_type = self.data_type, seed=test_seed)
        
        self.target_train = torch.from_numpy(np.argmax(self.target_train.astype('float32'), axis = 1))
        self.target_test = torch.from_numpy(np.argmax(self.target_test.astype('float32'), axis = 1))
        self.data_train = torch.from_numpy(self.data_train.astype('float32'),)
        self.data_test = torch.from_numpy(self.data_test.astype('float32'),)
        self.optimal_S_train = torch.from_numpy(self.optimal_S_train)
        self.optimal_S_test = torch.from_numpy(self.optimal_S_test)
       

        self.dataset_train = TensorDatasetAugmented(x = self.data_train, y = self.target_train, give_index = self.give_index)
        self.dataset_test = TensorDatasetAugmented(x = self.data_test, y = self.target_test, give_index = self.give_index)


class Syn1(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn1",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)


class Syn2(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn2",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)

class Syn3(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn3",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)

class Syn4(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn4",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)
            
class Syn5(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn5",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)

class Syn6(Syn_init):
    def __init__(self,
                mean = torch.tensor(0.0, dtype=torch.float32), 
                cov = torch.tensor(1.0, dtype=torch.float32),
                covariance_type = 'spherical',
                nb_sample_train = 10000,
                nb_sample_test = 10000,
                epsilon_sigma = 1.0,
                scaling_regression = True,
                give_index = False,
                noise_function = None,
                data_type = "Syn6",
                **kwargs):
        
        super().__init__(mean = mean,
                cov=cov,
                covariance_type = covariance_type,
                nb_sample_train = nb_sample_train,
                nb_sample_test = nb_sample_test,
                epsilon_sigma = epsilon_sigma,
                scaling_regression = scaling_regression,
                give_index = give_index,
                noise_function = noise_function,
                data_type=data_type,
                **kwargs)