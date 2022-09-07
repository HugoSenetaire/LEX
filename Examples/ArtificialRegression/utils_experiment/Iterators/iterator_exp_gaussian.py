
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_gmm



class GaussianIterator():
    def __init__(self, mean_imputation = False, list_component = [20,50,100]) -> None:
        
        self.list_component = list_component
        self.mean_imputation = mean_imputation



    def __iter__(self, args, dataset, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(args.args_output.folder, "weights")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            imputation_network_weights_path = os.path.join(folder_weight,dataset_name + "_" + str(component))
            if not os.path.exists(imputation_network_weights_path) :
                if hasattr(dataset, "data_train"):
                    train_gmm(dataset.data_train, component, imputation_network_weights_path)
                else :
                    data_train = torch.stack([dataset.dataset_train.__getitem__(k)[0] for k in range(len(dataset.dataset_train))])
                    train_gmm(data_train, component, imputation_network_weights_path)
            args.args_classification.module_imputation_parameters = {"imputation_network_weights_path": imputation_network_weights_path, "nb_component": component, "mean_imputation": self.mean_imputation}
            args.args_classification.module_imputation ="GaussianMixtureImputation"
            yield imputation_network_weights_path
        