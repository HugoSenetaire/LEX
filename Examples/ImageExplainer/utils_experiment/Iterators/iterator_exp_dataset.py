
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_gmm



class DatasetSamplingImputationIterator():
    def __init__(self, ) -> None:
        pass
        

    def get_data_val(self, dataset):
        if hasattr(dataset, "data_val"):
            return dataset.data_val
        elif hasattr(dataset, "dataset_val"):
            return torch.stack([dataset.dataset_val.__getitem__(k)[0] for k in range(len(dataset.dataset_train))])
        else :
            raise ValueError("Dataset has no data_val or dataset_val attribute")



    def __iter__(self, args, dataset, dataset_name):
        data_val = self.get_data_val(dataset)
            
        args.args_classification.module_imputation_parameters = {"data_to_impute": data_val,}
        args.args_classification.module_imputation ="DatasetSamplingImputation"
        yield dataset_name
        