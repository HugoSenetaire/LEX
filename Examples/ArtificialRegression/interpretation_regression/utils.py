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
from matplotlib.cm import ScalarMappable


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


global_keys = ["accuracy_selection_pi", "accuracy_selection_z", "accuracy_selection_thresholded_pi","accuracy_selection_thresholded_z",
 "proportion_pi", "proportion_z", "proportion_thresholded_pi", "proportion_thresholded_z", "accuracy_prediction_no_destruction",
  "accuracy_prediction_destruction", "mean_pi_list", "pi_list_q1", "pi_list_q2", "pi_list_median","auc_score_pi", "auc_score_z"]

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

# def get_dic_experiment(output_dic, to_save_dic, experiment_id):
#     to_save_dic["accuracy_prediction_no_destruction_test"][experiment_id] = output_dic["test"]["correct"][-1]
#     to_save_dic["accuracy_prediction_destruction_test"][experiment_id] =  output_dic["test"]["correct_destruction"][-1]
#     to_save_dic["pi_list_median_test"][experiment_id] = output_dic["test"]["pi_list_median"][-1]
#     to_save_dic["mean_pi_list_test"][experiment_id] = output_dic["test"]["mean_pi_list"][-1]
#     to_save_dic["pi_list_q1_test"][experiment_id] = output_dic["test"]["pi_list_q1"][-1]
#     to_save_dic["pi_list_q2_test"][experiment_id] = output_dic["test"]["pi_list_q2"][-1]
#     return to_save_dic

def get_dic_experiment(output_dic, to_save_dic, experiment_id, nb_experiments):
  for key in output_dic["test"].keys():
    save_key = key+ "_test"
    if save_key not in to_save_dic.keys():
      to_save_dic[save_key] = np.zeros((nb_experiments),)
    to_save_dic[save_key][experiment_id] = output_dic["test"][key][-1]
  return to_save_dic

def plot_true_continuousshape(dataset, path, figsize=(15,5)):
  centroids = dataset.centroids

  X = dataset.X_train
  Y = dataset.Y_train


  min_x = torch.min(dataset.X_train[:,0])
  max_x = torch.max(dataset.X_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, 1000)
  min_x = torch.min(dataset.X_train[:,1])
  max_x = torch.max(dataset.X_train[:,1])
  linspace_seconddim = torch.linspace(min_x, max_x, 1000)

  grid_x, grid_y = torch.meshgrid(linspace_firstdim, linspace_seconddim)
  xaux1 = grid_x.reshape(1, -1)
  xaux2 = grid_y.reshape(1, -1)
  complete_X = torch.cat([xaux1, xaux2], dim=0).transpose(1,0)
  aux_Y = dataset.centroids_Y.reshape(1,-1).expand(complete_X.shape[0],-1)

  # First dimension needed :
  mask = torch.zeros_like(complete_X)
  mask_firstdim = mask.clone()
  mask_firstdim[:,0]=torch.ones_like(complete_X[:,1])
  complete_x_firstdim = complete_X * mask_firstdim
  dependency_firstdim = dataset.get_dependency(mask_firstdim, complete_x_firstdim, index=None, dataset_type = None)
  true_selection_firstdim = 2 * torch.abs(0.5 - torch.sum(dependency_firstdim * aux_Y, axis = -1)) # Get true selection, 1 means selection; 0 means it's not necessary


  # Second dimension needed :
  mask_seconddim = mask.clone()
  mask_seconddim[:,1]=torch.ones_like(complete_X[:,1])
  complete_x_seconddim = complete_X * mask_seconddim
  dependency_seconddim = dataset.get_dependency(mask_seconddim, complete_x_seconddim, index=None, dataset_type = None)
  true_selection_seconddim = 2 * torch.abs(0.5 - torch.sum(dependency_seconddim * aux_Y, axis=-1))

  # All dim needed :
  aux = torch.cat([true_selection_firstdim.reshape(1,-1), true_selection_seconddim.reshape(1, -1)], dim=0) # Get true selection, 1 means selection; 0 means it's not necessary
  # true_selection_alldim = torch.max(aux,dim=0)[0] - torch.min(aux, dim=0)[0] # Get true selection, 1 means selection; 0 means it's not necessary
  true_selection_alldim = 1 - torch.max(aux, dim=0)[0]

  fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (15,5))
  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                  int(max(Y) + 1))))

  axs[0].contourf(grid_x, grid_y, true_selection_firstdim.reshape(grid_x.shape),  vmin=0, vmax=1.0)
  axs[1].contourf(grid_x, grid_y, true_selection_seconddim.reshape(grid_x.shape), vmin=0, vmax=1.0)
  qcs = axs[2].contourf(grid_x, grid_y, true_selection_alldim.reshape(grid_x.shape), vmin=0, vmax=1.0)

  for k in range(3):
    axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])

  fig.colorbar(
   ScalarMappable(norm=qcs.norm, cmap=qcs.cmap),
   ticks=range(0, 1, 20) 
  )
  complete_path = os.path.join(path, "dataset_selection.jpg")
  plt.savefig(complete_path)
  plt.close(fig)


def plot_destructor_output(destructor, dataset, path, figsize = (15,5)):
  try :
    centroids = dataset.centroids
  except :
    centroids = None
  X = dataset.X_train
  Y = dataset.Y_train


  min_x = torch.min(dataset.X_train[:,0])
  max_x = torch.max(dataset.X_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, 100)
  min_x = torch.min(dataset.X_train[:,1])
  max_x = torch.max(dataset.X_train[:,1])
  linspace_seconddim = torch.linspace(min_x, max_x, 100)

  grid_x, grid_y = torch.meshgrid(linspace_firstdim, linspace_seconddim)
  xaux1 = grid_x.reshape(1, -1)
  xaux2 = grid_y.reshape(1, -1)
  complete_X = torch.cat([xaux1, xaux2], dim=0).transpose(1,0)

  if next(destructor.destructor.parameters()).is_cuda:
    complete_X = complete_X.cuda()
  log_pi_list, _ = destructor(complete_X)
  pi_list = torch.exp(log_pi_list.detach().cpu())
  # First dimension needed :
  
  selection_firstdim = pi_list[:,0] # Get true selection, 1 means selection; 0 means it's not necessary


  # Second dimension needed :
  selection_seconddim = pi_list[:,1]

  # All dim needed :
  # aux = torch.cat([selection_firstdim.reshape(1,-1), selection_seconddim.reshape(1, -1)], dim=0) # Get true selection, 1 means selection; 0 means it's not necessary
  # selection_alldim = 1 - torch.max(aux, dim=0)[0]

  fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (10,5))
  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                  int(max(Y) + 1))))

  axs[0].contourf(grid_x, grid_y, selection_firstdim.reshape(grid_x.shape),  vmin=0, vmax=1.0)
  axs[1].contourf(grid_x, grid_y, selection_seconddim.reshape(grid_x.shape), vmin=0, vmax=1.0)
  # qcs = axs[2].contourf(grid_x, grid_y, selection_alldim.reshape(grid_x.shape), vmin=0, vmax=1.0)
  if centroids is not None :
    for k in range(2):
      axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])

  # fig.colorbar(
  #  ScalarMappable(norm=qcs.norm, cmap=qcs.cmap),
  #  ticks=range(0, 1, 20) 
  # )

  complete_path = os.path.join(path, "selector_selection.jpg")
  plt.savefig(complete_path)
  plt.close(fig)

def plot_model_output(trainer_var, dataset, sampling_distribution, path):
  try :
    centroids = dataset.centroids
  except :
    centroids = None
  X = dataset.X_train
  Y = dataset.Y_train


  min_x = torch.min(dataset.X_train[:,0])
  max_x = torch.max(dataset.X_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, 100)
  min_x = torch.min(dataset.X_train[:,1])
  max_x = torch.max(dataset.X_train[:,1])
  linspace_seconddim = torch.linspace(min_x, max_x, 100)

  grid_x, grid_y = torch.meshgrid(linspace_firstdim, linspace_seconddim)
  xaux1 = grid_x.reshape(1, -1)
  xaux2 = grid_y.reshape(1, -1)
  complete_X = torch.cat([xaux1, xaux2], dim=0).transpose(1,0)

  
  if next(trainer_var.classification_module.classifier.parameters()).is_cuda:
    complete_X = complete_X.cuda()

  try :
    pi_list, log_pi_list, _, z, p_z = trainer_var._destructive_test(complete_X, sampling_distribution, 1)
    log_y_hat_destructed, _ = trainer_var.classification_module(complete_X, z, index = None)
    pred_destruction = torch.exp(log_y_hat_destructed[:,0]).detach().cpu()
    destructive= True
  except(AttributeError) :
    destructive = False


  log_y_hat, _ = trainer_var.classification_module(complete_X, index = None)


  pred_classic = torch.exp(log_y_hat[:,0]).detach().cpu()



  # All dim needed :
  # aux = torch.cat([selection_firstdim.reshape(1,-1), selection_seconddim.reshape(1, -1)], dim=0) # Get true selection, 1 means selection; 0 means it's not necessary
  # selection_alldim = 1 - torch.max(aux, dim=0)[0]

  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                    int(max(Y) + 1))))
  if destructive :
    fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (10,5))
    axs[0].contourf(grid_x, grid_y, pred_classic.reshape(grid_x.shape),  vmin=0, vmax=1.0)
    axs[1].contourf(grid_x, grid_y, pred_destruction.reshape(grid_x.shape), vmin=0, vmax=1.0)
    # qcs = axs[2].contourf(grid_x, grid_y, selection_alldim.reshape(grid_x.shape), vmin=0, vmax=1.0)
    if centroids is not None:
      for k in range(2):
        axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])

  else :
    fig, axs = plt.subplots(nrows =1, ncols = 1, figsize = (5,5))
    axs.contourf(grid_x, grid_y, pred_classic.reshape(grid_x.shape),  vmin=0, vmax=1.0)

    if centroids is not None:
        axs.scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])

  complete_path = os.path.join(path, "output_classification.jpg")
  plt.savefig(complete_path)
  plt.close(fig)


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


def normalize_dic(dic, dic_count_batch_size, dic_count_dim_batch_size,):
  for key in dic_count_dim_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_dim_batch_size[key][k]
  for key in dic_count_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_batch_size[key][k]
  
  return dic