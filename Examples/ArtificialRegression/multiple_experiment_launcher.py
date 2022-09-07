import os
import sys
current_file_path = os.path.abspath(__file__)
while(not current_file_path.endswith("MissingDataTraining")):
    current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)


import traceback
from args_class import CompleteArgs
from missingDataTrainingModule.utils import compare_args, from_args_to_dictionary, dic_to_line_str
import missingDataTrainingModule
from datasets import *
from interpretation_regression import calculate_score, plot_complete_model_output, plot_selector_output

import matplotlib.pyplot as plt
from torch.distributions import *
from torch.optim import *
import torch
import pickle as pkl
import numpy as np
from utils_experiment import check_experiment_value


def get_dataset(args):
    args_dataset = args.args_dataset
    #Turn class to dictionary
    dic_parameters_dataset = vars(args_dataset.args_dataset_parameters)
    dataset = list_dataset[args_dataset.dataset](**dic_parameters_dataset)
    args.args_dataset.dataset_input_dim = dataset.get_dim_input()
    args.args_dataset.dataset_output_dim = dataset.get_dim_output()
    loader = LoaderArtificial(dataset, batch_size_train=args_dataset.args_dataset_parameters.batch_size_train, batch_size_test=args_dataset.args_dataset_parameters.batch_size_test,)
    return dataset, loader



def multiple_experiment(
            count,
            dataset,
            loader,
            complete_args,
            name_modification = True,):

    default_args = CompleteArgs()
    if not compare_args(complete_args, default_args) :
        raise ValueError("The arguments are not the default ones")

    if name_modification :
        line_name = dic_to_line_str(from_args_to_dictionary(complete_args, to_str=True))
        hashed = str(hash(line_name))
        complete_args.args_output.path = os.path.join(complete_args.args_output.path, hashed)


    if check_experiment_value(complete_args.args_output.path):
        return count+1
    
    try :
        final_path, trainer, loader, dic_list = missingDataTrainingModule.main_launcher.experiment(dataset, loader, complete_args)
        interpretable_module = trainer.interpretable_module
        
        ## Interpretation                
        dic_interpretation = calculate_score(interpretable_module, loader, trainer, complete_args, CFindex = None)
        current_path = os.path.join(final_path, "interpretation.txt")
        with open(current_path, "w") as f:
            for key in dic_interpretation:
                f.write(f"{key} : {dic_interpretation[key]}\n")
        current_path = os.path.join(final_path, "interpretation.pkl")
        with open(current_path, "wb") as f :
            pkl.dump(dic_interpretation, f)
        
        if loader.dataset.dim_input ==2:
            if hasattr(interpretable_module, "selection_module"):
                plot_selector_output(interpretable_module.selection_module, loader.dataset, complete_args.args_output.path)
                plot_selector_output(interpretable_module.selection_module, loader.dataset, complete_args.args_output.path, train_data=True)
                plot_selector_output(interpretable_module.selection_module, loader.dataset, complete_args.args_output.path, interpretation= True)
                plot_selector_output(interpretable_module.selection_module, loader.dataset, complete_args.args_output.path, interpretation= True, train_data=True)
                
            plot_complete_model_output(interpretable_module, loader.dataset, Bernoulli, complete_args.args_output.path)
            plot_complete_model_output(interpretable_module, loader.dataset, Bernoulli, complete_args.args_output.path, train_data=True)  

            out_path = os.path.join(complete_args.args_output.path, "output_dataset.png")
            Y = loader.dataset.target_train[:10000].cpu().detach().numpy()
            plt.scatter(loader.dataset.X[:10000,0], loader.dataset.X[:10000,1], c =Y, alpha = 0.15, cmap = 'gray', vmin = np.min(Y), vmax = np.max(Y))
            plt.savefig(out_path)
            plt.close()
            
        del trainer, loader, dic_list
        try :
            torch.cuda.empty_cache()
        except BaseException as e:
            print(e)
            print("Can't empty the cache")
        return count+1
    except Exception as e :
        print(traceback.format_exc())
        if os.path.exists(complete_args.args_output.path):
            os.rename(complete_args.args_output.path, complete_args.args_output.path+"_error")
        else :
            if not os.path.exists(complete_args.args_output.path+"_error"):
                os.makedirs(complete_args.args_output.path+"_error")
        with open(complete_args.args_output.path+"_error/error.txt", "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        
        try :
            torch.cuda.empty_cache()
        except BaseException as e:
            print(e)
            print("Can't empty the cache")
        return count+1

