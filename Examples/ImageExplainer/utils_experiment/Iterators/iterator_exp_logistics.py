
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
                list_component = [100],
                nb_epoch = 5,
                batch_size = 64,
                lr = 1e-4,
                nb_e_step = 10,
                nb_m_step = 10,
                type_of_training = "sgd",
                mean_imputation = False,):
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.list_component = list_component
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.nb_e_step = nb_e_step
        self.nb_m_step = nb_m_step
        self.type_of_training = type_of_training
        self.mean_imputation = mean_imputation



    def __iter__(self, args, loader, dataset_name):
        for component in self.list_component :
            folder_weight = os.path.join(args.args_output.folder, "weights")
            if not os.path.exists(folder_weight):
                os.makedirs(folder_weight)
            model_dir = os.path.join(folder_weight,"logistic_"+dataset_name + "_" + str(component)+ "_type_of_training_"+self.type_of_training)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(os.path.join(model_dir, "mixture_of_logistics.pt")):
                train_MixtureOfLogistics(loader,
                                        model_dir,
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
                                                                    "model_dir": model_dir,
                                                                    "nb_component": component,
                                                                    "transform_mean": self.transform_mean,
                                                                    "transform_std": self.transform_std,
                                                                    "type_of_training": self.type_of_training,
                                                                    "nb_e_step": self.nb_e_step,
                                                                    "nb_m_step": self.nb_m_step,
                                                                    "mean_imputation": self.mean_imputation,
                                                                    }
            args.args_classification.module_imputation ="MixtureOfLogisticsImputation"
            yield model_dir
        