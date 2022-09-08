
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_MixtureOfLogistics



class MixtureOfLogisticsIterator():
    def __init__(self,
                transform_mean,
                transform_std,
                list_component = [20,50,100],
                nb_epoch = 20,
                batch_size = 64,
                lr = 1e-4,
                nb_e_step = 10,
                nb_m_step = 10,
                type_of_training = "sgd"):
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.list_component = list_component
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.nb_e_step = nb_e_step
        self.nb_m_step = nb_m_step
        self.type_of_training = type_of_training



    def __iter__(self, args, loader, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(args.args_output.folder, "weights")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            imputation_network_weights_path = os.path.join(folder_weight,"logistic_"+dataset_name + "_" + str(component)+ "_type_of_training_"+self.type_of_training+".pth")
            if not os.path.exists(imputation_network_weights_path) :
                train_MixtureOfLogistics(loader,
                                        imputation_network_weights_path,
                                        component,
                                        epochs = self.nb_epoch,
                                        transform_mean = self.transform_mean,
                                        transform_std = self.transform_std,
                                        batch_size = self.batch_size,
                                        lr = self.lr,
                                        type_of_training = self.type_of_training,
                                        nb_e_step = self.nb_e_step,
                                        nb_m_step = self.nb_m_step)
            input_size = loader.dataset.get_dim_input()
            args.args_classification.module_imputation_parameters = {"input_size" : input_size,
                                                                    "imputation_network_weights_path": imputation_network_weights_path,
                                                                    "nb_component": component,
                                                                    "transform_mean": self.transform_mean,
                                                                    "transform_std": self.transform_std,
                                                                    "type_of_training": self.type_of_training,
                                                                    "nb_e_step": self.nb_e_step,
                                                                    "nb_m_step": self.nb_m_step,
                                                                    }
            args.args_classification.module_imputation ="MixtureOfLogisticsImputation"
            yield imputation_network_weights_path
        