import os
import sys
current_file_path = os.path.abspath(__file__)
while(not current_file_path.endswith("MissingDataTraining")):
    current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)

import argparse
import glob
import pandas as pd
import numpy as np
import tqdm
import pickle as pkl
import args_class
from missingDataTrainingModule import instantiate, load_full_module
import torch

from multiple_experiment_launcher import get_dataset
from interpretation_image import complete_analysis_image



def get_all_paths(input_dirs, dataset_name):
    list_all_paths = {}
    for input_dir in input_dirs :
        if dataset_name == "none" or dataset_name is None :
            possible_dataset = os.listdir(input_dir)
        else :
            possible_dataset = [dataset_name]
        for dataset in possible_dataset :
            if dataset not in list(list_all_paths.keys()) :
                list_all_paths[dataset] = []
            first_step = os.path.join(os.path.join(input_dir, dataset), "*")
            path_finder = os.path.join(os.path.join(first_step, "*"),"interpretation.txt")
            second_step = os.path.join(first_step, "*")
            path_finder_2 = os.path.join(os.path.join(second_step, "*"),"interpretation.txt")
            list_all_paths[dataset].extend(glob.glob(path_finder, recursive=True))
            list_all_paths[dataset].extend(glob.glob(path_finder_2, recursive=True))
            print("Found {} interpretations for dataset {}".format(len(list_all_paths[dataset]), dataset))
    print("Found {} paths".format(len(list_all_paths)))

    list_all_paths_new = {}
    for dataset_name in list_all_paths :
        list_all_paths_new[dataset_name] = []
        for path in list_all_paths[dataset_name] :
            list_all_paths_new[dataset_name] += [os.path.dirname(path)]
    return list_all_paths_new

def change_save_folder(path, input_dir):
    current_path = path.replace(input_dir, os.path.join(input_dir, "interpretation"))
    if not os.path.exists(current_path) :
        os.makedirs(current_path)
    return current_path


def get_parameters(path):
    parameter_path = os.path.join(os.path.join(path, "parameters"), "parameters.pkl")
    print(parameter_path)
    with open(parameter_path, "rb") as f :
        complete_args = pkl.load(f)
    
    return complete_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str, nargs='+',)
    parser.add_argument('--nb_samples_image_per_category', type=int, default=10)
    parser.add_argument('--nb_imputation', type=int, default=3)

    
    args = parser.parse_args()
    args_dict = vars(args)
    
    list_all_paths = get_all_paths(args.input_dirs, None)
    for dataset_path in list_all_paths :
        for folder_path in list_all_paths[dataset_path] :
            complete_args = get_parameters(folder_path)
            dataset, loader = get_dataset(complete_args)
            interpretable_module, complete_args_converted = instantiate(complete_args, dataset)
            final_path = change_save_folder(folder_path, args.input_dirs[0])
            if not os.path.exists(final_path) :
                os.makedirs(final_path)
            complete_args.args_output.path = final_path
            interpretable_module = load_full_module(folder_path, interpretable_module, suffix = "_last")
            if torch.cuda.is_available() :
                interpretable_module = interpretable_module.to("cuda:0")

            dic_interpretation = complete_analysis_image(interpretable_module, loader, None, args= complete_args, batch_size = 64, nb_samples_image_per_category = args.nb_samples_image_per_category, nb_imputation = args.nb_imputation)
            
            
            current_path = os.path.join(final_path, "interpretation.txt")
            with open(current_path, "w") as f:
                for key in dic_interpretation:
                    f.write(f"{key} : {dic_interpretation[key]}\n")
            current_path = os.path.join(final_path, "interpretation.pkl")
            with open(current_path, "wb") as f :
                pkl.dump(dic_interpretation, f)

