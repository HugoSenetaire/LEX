import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")

import os
import numpy as np
import torch
import torch.nn as nn
import sklearn
import torchvision
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pickle as pkl
from datasets import *
from default_parameter import *

import tqdm


noise_std = 0.2
n_components_list = [2, 20, 50, 100]
if __name__=="__main__":
    args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler = get_default()
    dataset = MNIST_and_FASHIONMNIST(args_dataset["root_dir"], download = False).dataset_train
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=8)

    data = dataset.data
    data = data.reshape(-1, 784*2)
    data = data.numpy().astype(np.float64)
    data += np.random.normal(0, noise_std, size = data.shape)

    dir_name = os.path.dirname(args_classification["imputation_network_weights_path"])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for n_components in tqdm.tqdm(n_components_list) :
        print(n_components)
        gm = GaussianMixture(n_components=n_components, covariance_type='diag',)

        save_path = os.path.join(os.path.dirname(args_classification["imputation_network_weights_path"]), f"{n_components}_components.pkl")
        
        gm.fit(data)
        mu = gm.means_
        covariances = gm.covariances_
        weights = gm.weights_
        pkl.dump((weights, mu, covariances), open(save_path, "wb"))
        print("save at ", save_path)
        print(f"{n_components} components saved")