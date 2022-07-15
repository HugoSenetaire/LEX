from numpy.lib.function_base import interp
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

sec_structure = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']





amino_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def interpretation_protein(path, samples, interpretations, targets, outputs, suffix = "", target_sequence = None):
    """ Draw the interpretation of the protein """

    samples = np.transpose(samples, axes= (0,2,1))

    fig, ax = plt.subplots(1,1)
    aux_output = np.array([k for k in range(len(outputs[0]))])
    aux_output = aux_output.reshape(-1,1)
    plt.imshow(aux_output, extent=[0, len(outputs[0]), 0, len(outputs[0])+1])
    ax.set_xticks(list(aux_output.flatten()))
    ax.set_xticklabels(sec_structure)
    plt.savefig(os.path.join(path, "legend.jpg"))
    plt.close(fig)

    mean_output = {}
    repartition_output = {}

    for i, (sample, interpretation, target, output) in enumerate(zip(samples, interpretations, targets, outputs)) :
        
        fig, axs = plt.subplots(1,2)

        path_target = os.path.join(path, f"target_{sec_structure[target]}")

        if not os.path.exists(path_target):
            os.makedirs(path_target)
        path_sample = os.path.join(path_target,f"sample_{i}")
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)


        aux = np.zeros(np.shape(interpretation))
        aux[0,len(aux[0])//2] = 1
        if target_sequence is not None :
            complete_sequence = np.argmax(target_sequence[i],-1)/len(output)
            complete_sequence = complete_sequence.reshape(1,-1)
            complete = np.concatenate([np.transpose(sample), interpretation, aux, complete_sequence])
        else :
            complete = np.concatenate([np.transpose(sample), interpretation, aux])


        if target.item() not in mean_output.keys():
            mean_output[target.item()] = complete
            repartition_output[target.item()] = 1
        else :
            mean_output[target.item()] += complete
            repartition_output[target.item()] +=1

        axs[0].imshow(complete, cmap='gray')
        x_pos = np.arange(0, len(output))
        axs[1].bar(x_pos, output)
        axs[1].set_xticks(x_pos, minor=False)
        axs[1].set_xticklabels(sec_structure, fontdict=None, minor=False)

        path_output = os.path.join(path_sample, f"interpretation_{suffix}.jpg")
        plt.savefig(path_output)
        plt.close(fig)


    for key in mean_output.keys():
        path_target = os.path.join(path, f"target_{sec_structure[key]}")
        path_mean = os.path.join(path_target, "MEAN_RESULTS_target.jpg")
        fig = plt.figure("Mean Results")
        plt.imshow(mean_output[key]/repartition_output[key])
        plt.savefig(path_mean)
        plt.close(fig)

    path_repartition = os.path.join(path, f"target_repartition.jpg")
    fig = plt.figure("Repartition")
    heights = []
    for k in range(len(output)) :
        if k in repartition_output.keys():
            heights.append(repartition_output[k])
        else :
            heights.append(0)
    plt.bar(
        np.arange(0,len(output),1),
        heights,
        tick_label = sec_structure,
        )
    plt.savefig(path_repartition)
    plt.close(fig)



    

            
def interpretation_protein_output_selected(path, samples, interpretations, targets, outputs, target_sequence = None):
    """ Draw the interpretation of the protein """
    samples = np.transpose(samples, axes= (0,2,1))

    fig, ax = plt.subplots(1,1)
    aux_output = np.array([k for k in range(len(outputs[0]))])
    aux_output = aux_output.reshape(-1,1)
    plt.imshow(aux_output, extent=[0, len(outputs[0]), 0, len(outputs[0])+1])
    ax.set_xticks(list(aux_output.flatten()))
    ax.set_xticklabels(sec_structure)
    plt.savefig(os.path.join(path, "legend.jpg"))
    plt.close(fig)

    mean_output = {}
    repartition_output = {}

    for i, (sample, interpretation, target, output) in enumerate(zip(samples, interpretations, targets, outputs)) :
        fig, axs = plt.subplots(1,2)

        detected = np.argmax(output, -1)
        path_detected = os.path.join(path, f"predicted_{sec_structure[detected]}")
        if not os.path.exists(path_detected):
            os.makedirs(path_detected)
        path_sample = os.path.join(path_detected,f"sample_{i}")
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)

        aux = np.zeros(np.shape(interpretation))
        aux[0,len(aux[0])//2] = 1
        if target_sequence is not None :
            complete_sequence = np.argmax(target_sequence[i],-1)/len(output)
            complete_sequence = complete_sequence.reshape(1,-1)  
            complete = np.concatenate([np.transpose(sample), interpretation, aux, complete_sequence])
        else :
            complete = np.concatenate([np.transpose(sample), interpretation, aux])

        if detected not in mean_output.keys():
            mean_output[detected] = complete
            repartition_output[detected] = 1
        else :
            mean_output[detected] += complete
            repartition_output[detected] +=1


        axs[0].imshow(complete, cmap='gray')
        x_pos = np.arange(0, len(output))
        axs[1].bar(x_pos, output)
        axs[1].set_xticks(x_pos, minor=False)
        axs[1].set_xticklabels(sec_structure, fontdict=None, minor=False)
        path_output = os.path.join(path_sample, "interpretation.jpg")
        plt.savefig(path_output)
        plt.close(fig)

    print("mean output keys", repartition_output.keys(), repartition_output)
    for key in mean_output.keys():
        path_target = os.path.join(path, f"predicted_{sec_structure[key]}")
        path_mean = os.path.join(path_target, "MEAN_RESULTS_DETECTED.jpg")
        fig = plt.figure("Mean Results Detected")
        plt.imshow(mean_output[key]/repartition_output[key])
        plt.savefig(path_mean)
        plt.close(fig)

    path_repartition = os.path.join(path, f"detected_repartition.jpg")
    fig = plt.figure("Repartition")
    heights = []
    for k in range(len(output)) :
        if k in repartition_output.keys():
            heights.append(repartition_output[k])
        else :
            heights.append(0)
    plt.bar(
        np.arange(0,len(output),1),
        heights,
        tick_label = sec_structure,
        )
    plt.savefig(path_repartition)
    plt.close(fig)



# TODO : Add interpretation using the secondary structure for neighbooring amino acids

