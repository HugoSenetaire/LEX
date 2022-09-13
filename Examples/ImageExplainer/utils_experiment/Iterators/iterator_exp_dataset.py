
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
        if hasattr(dataset, "dataset_val"):
            return dataset.dataset_val
        else :
            raise ValueError("Dataset has no dataset_val attribute")



    def __iter__(self, args, dataset, dataset_name):
        data_val = self.get_data_val(dataset)
            
        args.args_classification.module_imputation_parameters = {"dataset_to_impute": data_val,}
        args.args_classification.module_imputation ="DatasetSamplingImputation"
        yield dataset_name
        