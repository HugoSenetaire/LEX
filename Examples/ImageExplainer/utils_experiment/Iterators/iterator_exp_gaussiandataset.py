
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_gmm



class GaussianDatasetIterator():
    def __init__(self, list_component = [20,50,100], nb_points = 1000) -> None:
        
        self.list_component = list_component
        self.nb_points = nb_points

    def get_data_val(self, dataset):
        if hasattr(dataset, "data_val"):
            return dataset.data_val
        elif hasattr(dataset, "dataset_val"):
            return torch.stack([dataset.dataset_val.__getitem__(k)[0] for k in range(len(dataset.dataset_train))])
        else :
            raise ValueError("Dataset has no data_val or dataset_val attribute")

    def get_random_nb_points(self, data_val):
        if self.nb_points > len(data_val):
            self.nb_points = len(data_val)
        random_index = np.random.choice(len(data_val), self.nb_points, replace=False)
        return data_val[random_index]

    def __iter__(self, args, dataset, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(args.args_output.folder, "weights")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            imputation_network_weights_path = os.path.join(folder_weight,dataset_name + "_" + str(component))
            data_val = self.get_data_val(dataset)
            if not os.path.exists(imputation_network_weights_path) :
                train_gmm(data_val, component, imputation_network_weights_path)
            data_to_impute = self.get_random_nb_points(data_val)
                
            args.args_classification.module_imputation_parameters = {"imputation_network_weights_path": imputation_network_weights_path,
                                                                     "nb_component": component,
                                                                     "data_to_impute" : data_to_impute}
            args.args_classification.module_imputation ="GaussianMixtureDatasetImputation"
            yield imputation_network_weights_path
        