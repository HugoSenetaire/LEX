import os
import argparse
import glob
import pandas as pd
import numpy as np
import tqdm
import pickle as pkl

parameters_to_save = {}

# Train :
parameters_to_save["train.txt"] = ['nb_epoch', 'nb_epoch_pretrain_selector',
                                'nb_epoch_pretrain', 'nb_sample_z_train_monte_carlo',
                                'nb_sample_z_train_IWAE', 'fix_classifier_parameters',
                                'fix_selector_parameters', 'post_hoc', 'post_hoc_guidance',
                                'ratio_class_selection', "training_type",
                                "nb_step_fixed_classifier", "nb_step_fixed_selector", "nb_step_all_free",
                                "use_regularization_pretrain_selector",
                                "loss_function",
                                 ]

#Complete Trainer :
parameters_to_save["complete_trainer.txt"] = ['complete_trainer', 'monte_carlo_gradient_estimator',]

# classification module :
parameters_to_save["classification.txt"] = ['classifier', 'imputation',  'cste_imputation', 'nb_imputation_iwae', 'nb_imputation_iwae_test', 'nb_imputation_mc', 'nb_imputation_mc_test', 'post_process_sigma_noise']

# Selection module :
parameters_to_save["selection.txt"] = ['selector', 'regularization', 'lambda_reg', 'rate', 'loss_regularization', 'batched', 'activation']

# Distribution module :
parameters_to_save["distribution_module.txt"] = ["distribution_module", "distribution", "distribution_relaxed"]

# Classification Distribution module :
parameters_to_save["classification_distribution_module.txt"] = ["distribution_module", "distribution", "distribution_relaxed"]

# Dataset module :
parameters_to_save["dataset.txt"] = ["covariance_type", "dim_input", "used_dim"]



def get_all_paths(input_dirs, dataset_name):
    list_all_paths = []
    for input_dir in input_dirs :
        first_step = os.path.join(os.path.join(input_dir, dataset_name), "*")
        path_finder = os.path.join(os.path.join(first_step, "*"),"interpretation.txt")
        list_all_paths.extend(glob.glob(path_finder, recursive=True))

    print("Found {} paths".format(len(list_all_paths)))
    return list_all_paths

def read_interpretation(path, wanted_measure = ["accuracy", "accuracy_true_selection", "accuracy_no_selection", "auroc", "auroc_true_selection", "auroc_no_selection", "fpr2", "tpr2"]):
    dic = {}
    with open(path, "r") as f :
        text = f.readlines()
        for line in text :
            try :
                key, value = line.replace("\n", "").split(" : ")
                if key in wanted_measure :
                    dic[key] = float(value)
            except(ValueError):
                continue

    for key in wanted_measure :
        if key not in dic :
            dic[key] = None
            # if key ==   
            # TODO @hhjs : Interesting to get the value for all the mean and pi.   
    return dic

def parameter_to_dic(file):
    dic = {}
    with open(file, "r") as f :
        text = f.readlines()[0]
        text = text.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("<", "").replace(">", "").replace("\n", "")
        text = text.split(",")
        for k in range(len(text) -1, 0, -1):
            line = text[k]
            if ":" not in line:
                aux = text.pop(k)
                text[k-1] += "," + aux

        for line in text :
            key, value = line.split(": ")
            if value.startswith("class"):
                dic[key.replace(" ", '')] = value.split('.')[-1].replace(" ", '')
            else :
                dic[key.replace(" ", '')] = value.replace(" ", '')
    return dic


def get_parameters(path):
    complete_dic = {}
    folder = os.path.dirname(path)
    parameter_path = os.path.join(folder, "parameters")
    for key in parameters_to_save :
        path = os.path.join(parameter_path, key)
        value = parameters_to_save[key]
        
        if os.path.exists(path) :
            dic = parameter_to_dic(path)
            for k in value :
                if key == "classification_distribution_module.txt" :
                    k = "classification_"+k
                if k not in dic :
                    complete_dic[k] = "None"
                else :
                    complete_dic[k] = dic[k]
        else :
            print("NOT FOUND AT {}, {}, {}".format(key, value, path))
    return complete_dic

def output_to_dic(file):
    dic = {}
    with open(file, "r") as f :
        text = f.readlines()[0]
        text = text.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("<", "").replace(">", "").replace("\n", "")
        text = text.split(",")
        for k in range(len(text) -1, 0, -1):
            line = text[k]
            if ":" not in line:
                aux = text.pop(k)
                text[k-1] += "," + aux

        for line in text :
            key, value = line.split(": ")
            dic[key.replace(" ", '')] = value.replace(" ", '')
    return dic


def get_train_log(path,):
    complete_dic = {}
    folder = os.path.dirname(path)

    train_path = os.path.join(os.path.join(folder, "train"),"results_dic.pkl")
    test_path = os.path.join(os.path.join(folder, "test"), "results_dic.pkl")
    if os.path.exists(train_path) :
        with open(train_path, "rb") as f :
            dic_train = pkl.load(f)
        for key, value in dic_train.items():
            complete_dic[key + "train"] = np.mean(value[len(value)-10:])
    else :
        train_path = os.path.join(os.path.join(folder, "train"),"results_dic.txt")
        dic_train = output_to_dic(train_path)
        for key, value in dic_train.items():
            current_value = []
            splitted = value.split(',')
            for k in range(len(splitted)-1, len(splitted)-10, -1):
                current_value.append(float(splitted[k].replace(" ", '').replace("]", '')))
            complete_dic[key + "train"] = np.mean(current_value)
        if not os.path.exists(train_path):
            raise ValueError("Train path not found")
            
    # Parameter
    if os.path.exists(test_path) :
        with open(test_path, "rb") as f :
            dic_test = pkl.load(f)
        best_train_loss_in_test_index = np.argmin(dic_test["train_loss_in_test"]) 
        for key, value in dic_test.items():
            complete_dic[key + "test"] = "{:.5f}".format(value[-1]).replace(".", ",")
            complete_dic[key + "test_at_best_index"] = "{:.5f}".format(value[best_train_loss_in_test_index]).replace(".", ",")
    else :
        test_path = os.path.join(os.path.join(folder, "test"),"results_dic.txt")
        dic_test = output_to_dic(test_path)
        for key, value in dic_test.items():
            complete_dic[key + "test"] = value.split(',')[-1].replace(" ", '').replace("]", '')
        if not os.path.exists(test_path):
            raise ValueError
   
     
    return complete_dic


def create_data_frame(input_dirs, dataset_name, get_output = False):
    list_all_paths = get_all_paths(input_dirs, dataset_name)
    dataframe = None
    dic = {}
    for k, path in tqdm.tqdm(enumerate(list_all_paths)) :
        # Parameter
        dic = get_parameters(path)
        # Interpretation
        interpretation = read_interpretation(path)
        dic.update(interpretation)
        dic["path"] = path
        # Output
        if get_output :
            try :
                output = get_train_log(path)
            except(ValueError):
                print("Error at {}, file not found".format(path))
                continue
            dic.update(output)
        if k == 0 :
            # dataframe = pd.DataFrame(columns=[list(dic.keys())])
            dataframe = pd.DataFrame(dic, index=[k])
        else :
            dataframe = dataframe.append(dic, ignore_index=True)


                
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default="default",
                        help='filename for the results')
    parser.add_argument('--out_dir', type=str, default='exps',
                        help='directory name for results')
    parser.add_argument('--input_dirs', type=str, nargs='+',)
    parser.add_argument('--dataset_name', type = str, default = 'S1')
    parser.add_argument('--get_output',action = 'store_true', )
    
    args = parser.parse_args()
    args_dict = vars(args)

    assert len(args.input_dirs) > 0, "Please provide at least one input directory"

    df = create_data_frame(input_dirs = args.input_dirs, dataset_name=args.dataset_name, get_output = args.get_output)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = os.path.join(args.out_dir, args.out_path + '.csv')
    df.to_csv(out_file, sep=';')