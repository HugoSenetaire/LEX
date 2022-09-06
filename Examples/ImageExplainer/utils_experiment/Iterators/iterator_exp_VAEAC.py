
import os
import sys
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_file_path)
from default_parameter import *
from missingDataTrainingModule import train_VAEAC



class VAEACIterator():
    def __init__(self, model_dir_list,) -> None:
        if not isinstance(model_dir_list, list):
            self.model_dir_list = [model_dir_list]
        else :
            self.model_dir_list = model_dir_list
        
        for model_dir in self.model_dir_list :
            if not os.path.exists(model_dir):
                raise ValueError("The model directory does not exist")
            

    def __iter__(self, args, loader,):
        for model_dir in self.model_dir_list :
            if loader is not None:
                if not os.path.exists(os.path.join(model_dir, "last_checkpoint.tar")):
                    train_VAEAC(loader, model_dir, epochs = 50)
            args.args_classification.module_imputation_parameters = {"model_dir": model_dir, }
            args.args_classification.module_imputation = "VAEACImputation"
            yield model_dir
        