import torch
import matplotlib.pyplot as plt

from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt

import os
from datetime import datetime
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


    
def show_interpretation_tabular(sample, data, target, dataloader):

 
    for i in range(len(sample)):
        list_element = []
        x_list = []
        print(f"Wanted target category : {target[i]}")
        for index in range(np.shape(data)[1]):
            print("Category : {} \t Proba : {} \t Value : {}".format(
                dataloader.train_loader.dataset.get_coded_category(index),
                sample[i][index],
                data[i][index]
            ))
            list_element.append(dataloader.train_loader.dataset.get_coded_category(index))
            x_list.append(index)
        x_list = np.arange(len(list_element))
        width = 0.35
        fig, ax = plt.subplots()
        ax1 = ax.bar(x_list-width/2, sample[i], width, label = 'Sample')
        ax2 = ax.bar(x_list+width/2, data[i], width, label = 'Data')
        ax.set_xticks(x_list)
        ax.set_xticklabels(list_element, rotation = 45, ha="right", rotation_mode = "anchor")
        ax.legend()
        plt.show()

