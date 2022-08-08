import os
import argparse
import glob
import pandas as pd
import numpy as np
import tqdm
import pickle as pkl
import args_class


def get_all_paths(input_dirs, dataset_name):
    list_all_paths = {}
    for input_dir in input_dirs :
        if dataset_name == "none" :
            possible_dataset = os.listdir(input_dir)
        else :
            possible_dataset = [dataset_name]
        for dataset in possible_dataset :
            if dataset not in list(list_all_paths.keys()) :
                list_all_paths[dataset] = []
            first_step = os.path.join(os.path.join(input_dir, dataset), "*")
            path_finder = os.path.join(os.path.join(first_step, "*"),"interpretation.txt")
            list_all_paths[dataset].extend(glob.glob(path_finder, recursive=True))
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
    
    with open(file, "rb") as f :
        complete_args = pkl.load(f)

    for element_key in vars(complete_args).keys() :
        # dic.update(vars(getattr(complete_args, element_key)))
        aux_dic = vars(getattr(complete_args, element_key))
        aux_dic = {"parameters_" + element_key + "_" + key : val for key, val in aux_dic.items()}
        dic.update(aux_dic)
        # dic.update(vars(getattr(complete_args, element_key)))

    current_keys = list(dic.keys())
    for element_key in current_keys :
        try :
            _ = dic[element_key].keys()
            dic.update(dic[element_key])
            del dic[element_key]
        except :
            continue
    
    for key in dic.keys():
        try :
            dic[key] = dic[key].item()
        except AttributeError as e:
            continue
    
    for key in dic.keys():
        if dic[key] == None :
            dic[key] = "None"

    for key in dic.keys():
        try :
            if len(dic[key]) > 0 :
                dic[key] = str(dic[key])
        except TypeError as e :
            continue

    dic["parameters_args_dataset_dataset"] = str(dic["parameters_args_dataset_dataset"])
    dic["parameters_args_dataset_loader"] = str(dic["parameters_args_dataset_loader"])


    return dic


def get_parameters(path):
    complete_dic = {}
    folder = os.path.dirname(path)
    parameter_path = os.path.join(os.path.join(folder, "parameters"), "parameters.pkl")
    complete_dic = parameter_to_dic(parameter_path)
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

    # OUTPUT TEST
    test_path = os.path.join(os.path.join(folder, "test"), "results_dic.pkl")
    with open(test_path, "rb") as f :
        dic_test = pkl.load(f)
    try :
        best_train_loss_in_test_index = np.argmin(dic_test["train_loss_in_test"]) 
        for key, value in dic_test.items():
            try :
                if len(value[-1])>0 :
                    continue
                complete_dic[key + "test"] = value[-1]
                complete_dic[key + "test_at_best_index"] = value[best_train_loss_in_test_index]
            except :
                complete_dic[key + "test"] = value[-1]
                complete_dic[key + "test_at_best_index"] = value[best_train_loss_in_test_index]

    except(KeyError):
        print("KeyError, train_loss_in_test not found for {}".format(path))
        for key, value in dic_test.items():
            if len(value[-1])>0 :
                continue
            complete_dic[key + "test"] = value[-1]
                
   
     
    return complete_dic


def create_data_frame(input_dirs, dataset_name, get_output = False):
    list_all_paths = get_all_paths(input_dirs, dataset_name)
    dataframe = None
    dic = {}
    k=0
    for dataset_name in list_all_paths :
        print("Treating {}".format(dataset_name))
        for i,path in tqdm.tqdm(enumerate(list_all_paths[dataset_name])) :
            # Parameter
            dic = get_parameters(path)
            # Interpretation
            interpretation = read_interpretation(path)
            dic.update(interpretation)
            # Output
            if get_output :
                try :
                    output = get_train_log(path)
                except(ValueError):
                    print("Error at {}, file not found".format(path))
                    continue
                dic.update(output)
            if k == 0 :
                dataframe = pd.DataFrame(dic, index=[k])
            else :
                dataframe = dataframe.append(dic, ignore_index=True)
            k+=1

    return dataframe

def get_average_and_std(df):
    list_keys = list(df.keys())
    list_parameters = []
    list_measure = []
    for key in list_keys :
        if key.startswith("parameters_") :
            if key.startswith("parameters_args_dataset_") :
                continue
            elif key.startswith("parameters_args_output_") :
                continue
            else :
                list_parameters.append(key)
        else :
            list_measure.append(key)

    df = df[list_parameters + list_measure]
    dic_change = {list_measure[i] : ['mean', 'std'] for i in range(len(list_measure))}
    list_measure_name = []
    for name in list_measure :
        list_measure_name.append(name + "_mean")
        list_measure_name.append(name + "_std") 
    df_grouped = df.groupby(list_parameters).agg(dic_change).reset_index()
    df_grouped.columns = list_parameters + list_measure_name
    
    return df_grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default="default",
                        help='filename for the results')
    parser.add_argument('--out_dir', type=str, default='exps',
                        help='directory name for results')
    parser.add_argument('--input_dirs', type=str, nargs='+',)
    parser.add_argument('--dataset_name', type = str, default = 'none')
    parser.add_argument('--get_output',action = 'store_true', )

    
    args = parser.parse_args()
    args_dict = vars(args)
    

    assert len(args.input_dirs) > 0, "Please provide at least one input directory"

    df = create_data_frame(input_dirs = args.input_dirs, dataset_name=args.dataset_name, get_output = args.get_output)
    df_grouped = get_average_and_std(df)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = os.path.join(args.out_dir, args.out_path + '.csv')
    df.to_csv(out_file, sep=';')

    out_file_grouped = os.path.join(args.out_dir, args.out_path + '_grouped.csv')
    df_grouped.to_csv(out_file_grouped, sep=';')
