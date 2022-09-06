
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from multiple_experiment_launcher import get_dataset
from missingDataTrainingModule import train_gmm_latent, load_full_module, instantiate
import pickle as pkl


class GaussianLatentIterator():
    def __init__(self, path_module = r"C:\Users\hhjs\Documents\FirstProject\MissingDataTraining\Experiments\weights\MNIST_and_FASHIONMNIST_autoencoder") -> None:        
        self.list_component = [20, 50, 100,] 
        self.path_module = path_module

    def train_gmm(self, component, dataset, imputation_network_weights_path):
        parameters_path_module = os.path.join(os.path.join(self.path_module, "parameters"), "parameters.pkl")
        args_autoencoder = pkl.load(open(parameters_path_module, "rb"))
        interpretable_module = instantiate(args_autoencoder)
        interpretable_module = load_full_module(self.path_module, interpretable_module)
        autoencoder = interpretable_module.prediction_module
        if hasattr(dataset, "data_train"):
            train_gmm_latent(dataset.data_train, autoencoder, component, imputation_network_weights_path)
        else :
            data_train = torch.stack([dataset.dataset_train.__getitem__(k)[0] for k in range(len(dataset.dataset_train))])
            train_gmm_latent(data_train, autoencoder, component, imputation_network_weights_path)


    def __iter__(self, args, dataset, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(self.path_module, "weights_gaussian_latent")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            imputation_network_weights_path = os.path.join(folder_weight,dataset_name + "_" + str(component))
            if not os.path.exists(imputation_network_weights_path) :
                self.train_gmm( component, dataset, imputation_network_weights_path)
                
            args.args_classification.module_imputation_parameters = {"imputation_network_weights_path": imputation_network_weights_path, "nb_component": component, "path_module": self.path_module}
            args.args_classification.module_imputation = "GaussianMixtureLatentImputation"
            yield imputation_network_weights_path
        