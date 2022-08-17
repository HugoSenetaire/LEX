import os
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch


def fill_dic(total_dic, dic):
    if len(total_dic.keys())==0:
        for key in dic.keys():
            if isinstance(dic[key], list):
                total_dic[key]=dic[key]
            else :
                total_dic[key] = [dic[key]]
    else :
        for key in dic.keys():
            if isinstance(dic[key], list):
                total_dic[key].extend(dic[key])
            else :
                total_dic[key].append(dic[key])


    return total_dic

def save_dic(path, dic):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "results_dic.pkl"), "wb") as f:
        pkl.dump(dic, f)

    with open(os.path.join(path,"results_dic.txt"), "w") as f:
        f.write(str(dic))


    for key in dic.keys():
        if key == "epoch" or key.startswith("confusion_matrix"):
            continue
        table = dic[key]
        plt.figure(0)
        try : 
            epoch_indexes = dic["epoch"]
            plt.plot(epoch_indexes,table)
        except KeyError:
            plt.plot(np.linspace(0,len(table)-1,len(table)),table)

        plt.savefig(os.path.join(path,str(key)+".jpg"))
        plt.clf()



def from_args_to_dictionary(args, to_str = True):
    """
    Transform any args object to a dictionary.
    """
    dic = {}
    list_to_check = [("", args)]
    while len(list_to_check)>0:
        current_name, current_arg = list_to_check.pop()
        if isinstance(current_arg, dict):
            for key in current_arg.keys():
                list_to_check.append((str(current_name)+str(key)+"_", current_arg[key]))
        else :
            try :
                list_to_check.append((str(current_name), current_arg.__dict__))
            except AttributeError:
                if to_str :
                    dic[str(current_name)] = str(current_arg)
                else :
                    dic[current_name] = current_arg
    return dic

def dic_to_line_str(dic):
    """
    Transform a dictionary to a string.
    """
    line = ""
    for key in dic.keys():
        line += str(key) + " : " + str(dic[key]) + "\n"
    return line

def compare_args(default_args, created_args):
    """
    Compare two args objects and return a dictionary with the differences.
    Make sur we don't create unused keys in the network.
    """
    default_args_dic = from_args_to_dictionary(default_args)
    created_args_dic = from_args_to_dictionary(created_args)

    set_default = set(list(default_args_dic.keys()))
    set_created = set(list(created_args_dic.keys()))
    set_different = set_default.difference(set_created)
    for k in list(set_different) :
        if "optim" in k or "scheduler" in k :
            set_different.remove(k)
    if len(set_different)>0:
        print("The following arguments are missing in the created args :")
        print(set_different)
        return False
    else :
        return True


    
            
