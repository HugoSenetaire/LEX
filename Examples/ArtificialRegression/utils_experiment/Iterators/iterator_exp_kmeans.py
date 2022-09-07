
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_kmeans



class KmeansIterator():
    def __init__(self, list_component = [20,50,100], nb_points = 1000) -> None:
        
        self.list_component = list_component
        self.nb_points = nb_points

    def get_data_train(self, dataset):
        if hasattr(dataset, "data_train"):
            return dataset.data_train
        else :
            return torch.stack([dataset.dataset_train.__getitem__(k)[0] for k in range(len(dataset.dataset_train))])

    def get_random_nb_points(self, data_train):
        if self.nb_points > len(data_train):
            self.nb_points = len(data_train)
        random_index = np.random.choice(len(data_train), self.nb_points, replace=False)
        return data_train[random_index]

    def __iter__(self, args, dataset, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(args.args_output.folder, "weights")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            imputation_network_weights_path = os.path.join(folder_weight,dataset_name + "_kmeans_"+ str(component))

            data_train = self.get_data_train(dataset)
            if not os.path.exists(imputation_network_weights_path) :
                train_kmeans(data_train, component, imputation_network_weights_path)
            data_to_impute = self.get_random_nb_points(data_train)
            
            args.args_classification.module_imputation_parameters = {"imputation_network_weights_path": imputation_network_weights_path,
                                                                    "nb_component": component,
                                                                    "nb_points" : self.nb_points,
                                                                    "data_to_impute": data_to_impute, }
            args.args_classification.module_imputation ="KmeansDatasetImputation"
            yield imputation_network_weights_path
        