import torch
import matplotlib.pyplot as plt

from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
from itertools import cycle, islice

import os
from datetime import datetime
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00'])

def save_result_artificial(path, data, target, predicted):
  path_result = os.path.join(path, "result")

  if not os.path.exists(path_result):
    os.makedirs(path_result)

  nb_dim = data.shape[1]

  for dim1 in range(nb_dim-1):
    for dim2 in range(dim1+1, nb_dim):
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
      ax1.scatter(data[:,dim1],data[:, dim2], s=10, color = colors[target])
      ax1.set_title('Input Data')
      ax2.scatter(data[:,dim1],data[:, dim2], s=10, color = colors[predicted])
      ax2.set_title('Prediction')
      plt.savefig(os.path.join(path_result, f"Prediction_output_dim_{dim1}_{dim2}.jpg"))
      plt.close(fig)

def save_interpretation_artificial(path, data_destructed, target, predicted, prefix = ""):
  path_result = os.path.join(path, "result")
  if not os.path.exists(path_result):
    os.makedirs(path_result)

  nb_dim = data_destructed.shape[1]

  for dim1 in range(nb_dim):
    for dim2 in range(dim1+1, nb_dim):


      fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
      ax1.scatter(data_destructed[:,dim1], data_destructed[:,dim2], s=10, color = colors[target])
      ax1.set_title('Destructed data with target')
      ax2.scatter(data_destructed[:,dim1], data_destructed[:,dim2], s=10, color = colors[predicted])
      ax2.set_title('Destructed data with prediction')
      plt.savefig(os.path.join(path_result, f"{prefix}_Destructed_output_dim_{dim1}_{dim2}.jpg"))
      plt.close(fig)



def save_interpretation_artificial_bar(path, sample, target, pred):
  path_result = os.path.join(path, "result")
  if not os.path.exists(path_result):
    os.makedirs(path_result)

  nb_dim = sample.shape[1]


  sample_list = []
  label_list = []
  for dim in range(nb_dim):
    sample_list.append(sample[np.where(target==0),dim][0])
    label_list.append(f"Coord {dim} Y 0")
    sample_list.append(sample[np.where(target==1),dim][0])
    label_list.append(f"Coord {dim} Y 1")
    


  fig = plt.figure(1)
  plt.boxplot(sample_list,
     labels = label_list)

  plt.savefig(os.path.join(path_result, f"box_plot_real_target.jpg"))
  plt.close(fig)

  sample_list = []
  label_list = []
  for dim in range(nb_dim):
    sample_list.append(sample[np.where(pred==0),dim][0])
    label_list.append(f"Coord {dim} Y 0")
    sample_list.append(sample[np.where(pred==1),dim][0])
    label_list.append(f"Coord {dim} Y 1")
    

  fig = plt.figure(1)
  plt.boxplot(sample_list,
     labels = label_list)
  
  plt.savefig(os.path.join(path_result, f"box_plot_real_pred.jpg"))
  plt.close(fig)

### SAVINGS FOR HYPERCUDE DATASET


global_keys = ["accuracy_selection_pi", "accuracy_selection_z", "accuracy_selection_thresholded_pi","accuracy_selection_thresholded_z", "proportion_pi", "proportion_z", "proportion_thresholded_pi", "proportion_thresholded_z", "accuracy_prediction_no_destruction", "accuracy_prediction_destruction",
     "mean_pi_list", "pi_list_q1", "pi_list_q2", "pi_list_median","auc_score_pi", "auc_score_z"]

global_keys_to_count_batch_size = ["proportion_pi", "proportion_z", "proportion_thresholded_pi", "proportion_thresholded_z"]
global_keys_to_count_dim_batch_size = ["accuracy_selection_pi", "accuracy_selection_z", "accuracy_selection_thresholded_pi","accuracy_selection_thresholded_z","auc_score_pi", "auc_score_z"]



def create_dics(keys, nb_experiments, keys_to_count_batch_size, keys_to_count_dim_batch_size):
    dic = {}
    dic_count_batch_size = {}
    dic_count_dim_batch_size = {}
    for key in keys :
        dic[key+"_train"] = np.zeros((nb_experiments),)
        dic[key+"_test"] = np.zeros((nb_experiments),)
        if key in keys_to_count_batch_size :
            dic_count_batch_size[key+"_train"] = np.zeros((nb_experiments),)
            dic_count_batch_size[key+"_test"] = np.zeros((nb_experiments),)
        elif key in keys_to_count_dim_batch_size :
            dic_count_dim_batch_size[key+"_train"] = np.zeros((nb_experiments,),)
            dic_count_dim_batch_size[key+"_test"] = np.zeros((nb_experiments,),)
        
    return dic, dic_count_batch_size, dic_count_dim_batch_size

def get_dic_experiment(output_dic, to_save_dic, experiment_id):
    to_save_dic["accuracy_prediction_no_destruction_test"][experiment_id] = output_dic["test"]["correct"][-1]
    to_save_dic["accuracy_prediction_destruction_test"][experiment_id] =  output_dic["test"]["correct_destruction"][-1]
    to_save_dic["pi_list_median_test"][experiment_id] = output_dic["test"]["pi_list_median"][-1]
    to_save_dic["mean_pi_list_test"][experiment_id] = output_dic["test"]["mean_pi_list"][-1]
    to_save_dic["pi_list_q1_test"][experiment_id] = output_dic["test"]["pi_list_q1"][-1]
    to_save_dic["pi_list_q2_test"][experiment_id] = output_dic["test"]["pi_list_q2"][-1]
    return to_save_dic


def get_evaluation(trainer_var, loader, dic, dic_count_batch_size, dic_count_dim_batch_size, experiment_id, current_sampling_test, args_train, train = False):
    if train:
        suffix = "_train"
        wanted_loader = loader.train_loader
    else :
        suffix = "_test"
        wanted_loader = loader.test_loader

    dataset = loader.dataset
    for (data, target, index) in iter(wanted_loader):
        batch_size, dim = data.shape
        if args_train["use_cuda"]:
            data = data.cuda()
        pi_list, log_pi_list,  _, z_s, _ = trainer_var._destructive_test(data, current_sampling_test, 1)
        z_s = z_s.reshape(data.shape)
        comparing_dic = dataset.compare_selection(pi_list, index=index, normalized = False, train_dataset = True, sampling_distribution = current_sampling_test)
        

        for key in comparing_dic.keys():
            dic[key+suffix][experiment_id] += comparing_dic[key]

        for key in dic_count_batch_size.keys():
            dic_count_batch_size[key][experiment_id] += batch_size
        for key in dic_count_dim_batch_size.keys():
            dic_count_dim_batch_size[key][experiment_id] += dim * batch_size

    return dic, dic_count_batch_size, dic_count_dim_batch_size