import os
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl



def fill_dic(total_dic, dic):
    if len(total_dic.keys())==0:
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key]=dic[key]
            else :
                total_dic[key] = [dic[key]]
    else :
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key].extend(dic[key])
            else :
                total_dic[key].append(dic[key])


    return total_dic

def save_dic(path, dic):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "results_dic.pkl"), "wb") as f:
        pkl.dump(dic, f)

    for key in dic.keys():
        table = dic[key]
        plt.figure(0)
        plt.plot(np.linspace(0,len(table)-1,len(table)),table)
        plt.savefig(os.path.join(path,str(key)+".jpg"))
        plt.clf()

