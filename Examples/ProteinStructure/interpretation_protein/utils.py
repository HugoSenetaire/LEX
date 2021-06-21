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
amino_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']

def interpretation_proteinv1(path, samples, interpretations, targets, outputs):
    """ Draw the interpretation of the protein """

    samples = np.transpose(samples, axes= (0,2,1))
    print(samples.shape)

    # for i in range(len(samples)):
    #     print(i, np.count_nonzero(samples[0]), np.where(samples[0]>0)[0], np.where(samples[0]>0)[1])

    for i, (sample, interpretation, target, output) in enumerate(zip(samples, interpretations, targets, outputs)) :
        fig, axs = plt.subplots(2,2)

        path_target = os.path.join(path, f"target_{sec_structure[target]}")
        if not os.path.exists(path_target):
            os.makedirs(path_target)
        path_sample = os.path.join(path_target,f"sample_{i}")
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)

        axs[0,0].imshow(sample, cmap= 'gray')
        axs[1,0].imshow(interpretation, cmap='gray')

        gs = axs[1, 1].get_gridspec()
        # remove the underlying axes
        for ax in axs[1:, -1]:
            ax.remove()
        axbig = fig.add_subplot(gs[1:, -1])
        x_pos = np.arange(0, len(output))
        axbig.bar(x_pos, output)
        # axbig.xticks(ticks = x_pos, labels=sec_structure)
        axbig.set_xticks(x_pos, minor=False)
        axbig.set_xticklabels(sec_structure, fontdict=None, minor=False)
        path_output = os.path.join(path_sample, "interpretation.jpg")
        plt.savefig(path_output)
        plt.close(fig)

            



def interpretation_protein(path, samples, interpretations, targets, outputs):
    """ Draw the interpretation of the protein """

    samples = np.transpose(samples, axes= (0,2,1))


    for i, (sample, interpretation, target, output) in enumerate(zip(samples, interpretations, targets, outputs)) :
        fig, axs = plt.subplots(1,2)

        path_target = os.path.join(path, f"target_{sec_structure[target]}")
        if not os.path.exists(path_target):
            os.makedirs(path_target)
        path_sample = os.path.join(path_target,f"sample_{i}")
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)

        # axs[0,0].imshow(sample, cmap= 'gray')
        # axs[1,0].imshow(interpretation, cmap='gray')
        aux = np.zeros(np.shape(interpretation))
        aux[0,len(aux[0])//2] = 1
        complete = np.concatenate([np.transpose(sample), interpretation, aux])
        axs[0].imshow(complete, cmap='gray')
        # axs[1] = fig.add_subplot(gs[1:, -1])
        x_pos = np.arange(0, len(output))
        axs[1].bar(x_pos, output)
        # axs[1].xticks(ticks = x_pos, labels=sec_structure)
        axs[1].set_xticks(x_pos, minor=False)
        axs[1].set_xticklabels(sec_structure, fontdict=None, minor=False)
        path_output = os.path.join(path_sample, "interpretation.jpg")
        plt.savefig(path_output)
        plt.close(fig)

            
def interpretation_protein_global(path, samples, interpretations, targets, outputs):
    """ Draw the interpretation of the protein """

    samples = np.transpose(samples, axes= (0,2,1))


    for possible_target in len(outputs) :
        fig, axs = plt.subplots(1,2)

        path_target = os.path.join(path, f"target_{sec_structure[target]}")
        if not os.path.exists(path_target):
            os.makedirs(path_target)
        path_sample = os.path.join(path_target,f"sample_{i}")
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)

        # axs[0,0].imshow(sample, cmap= 'gray')
        # axs[1,0].imshow(interpretation, cmap='gray')
        aux = np.zeros(np.shape(interpretation))
        aux[0,len(aux[0])//2] = 1
        complete = np.concatenate([np.transpose(sample), interpretation, aux])
        axs[0].imshow(complete, cmap='gray')
        # axs[1] = fig.add_subplot(gs[1:, -1])
        x_pos = np.arange(0, len(output))
        axs[1].bar(x_pos, output)
        # axs[1].xticks(ticks = x_pos, labels=sec_structure)
        axs[1].set_xticks(x_pos, minor=False)
        axs[1].set_xticklabels(sec_structure, fontdict=None, minor=False)
        path_output = os.path.join(path_sample, "interpretation.jpg")
        plt.savefig(path_output)
        plt.close(fig)



