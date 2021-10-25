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




def plot_true_classifier_output(dataset, path, train_data = False, nb_train_data = 1000, figsize=(15,5)):
  centroids = dataset.centroids

  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]



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
  mask = torch.ones_like(complete_X)
  dependency = dataset.get_dependency(mask, complete_X, index=None, dataset_type = None)
  true_selection = torch.sum(dependency * aux_Y, axis = -1) 


  fig, axs = plt.subplots(nrows =1, ncols = 1, figsize = (5,5))
  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                  int(max(Y) + 1))))

  axs.contourf(grid_x, grid_y, true_selection.reshape(grid_x.shape),  vmin=0, vmax=1.0)
  

  axs.scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
  
  if train_data :
    axs.scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  complete_path = os.path.join(path, "true_classifier.jpg")

  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)


def plot_true_continuousshape(dataset, path, train_data = False, nb_train_data = 1000, interpretation = False, figsize=(15,5)):
  centroids = dataset.centroids

  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]



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
  mask_all_dim = torch.ones_like(complete_X)
  dependency_all_dim = dataset.get_dependency(mask_all_dim, complete_X, index=None, dataset_type = None)
  true_selection_alldim = 2 * torch.abs(0.5 - torch.sum(dependency_all_dim * aux_Y, axis=-1))
  aux = torch.cat([true_selection_firstdim.reshape(1,-1), true_selection_seconddim.reshape(1, -1)], dim=0) # Get true selection, 1 means selection; 0 means it's not necessary
  # true_selection_alldim = torch.max(aux,dim=0)[0] - torch.min(aux, dim=0)[0] # Get true selection, 1 means selection; 0 means it's not necessary

  if interpretation :
    true_selection_alldim = true_selection_alldim - torch.max(aux, dim=0)[0]
    true_selection_alldim = torch.where(true_selection_alldim<0., torch.zeros_like(true_selection_alldim), true_selection_alldim)
  else :
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
  
  if train_data :
    for k in range(3):
      axs[k].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  fig.colorbar(
   ScalarMappable(norm=qcs.norm, cmap=qcs.cmap),
   ticks=range(0, 1, 20) 
  )


  complete_path = os.path.join(path, "Y_definition.jpg")
  if interpretation :
    complete_path = complete_path.split(".jpg")[0] + "_interpretation.jpg"

  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)

def plot_true_interpretation_v2(dataset, path, train_data = False, nb_train_data = 1000,  figsize=(15,5)):
  centroids = dataset.centroids

  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]

  grid_sample = 1000

  min_x = torch.min(dataset.X_train[:,0])
  max_x = torch.max(dataset.X_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, grid_sample)
  min_x = torch.min(dataset.X_train[:,1])
  max_x = torch.max(dataset.X_train[:,1])
  linspace_seconddim = torch.linspace(min_x, max_x, grid_sample)

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
  Y_first_dim = torch.sum(dependency_firstdim * aux_Y, axis = -1)

  # Second dimension needed :
  mask_seconddim = mask.clone()
  mask_seconddim[:,1]=torch.ones_like(complete_X[:,1])
  complete_x_seconddim = complete_X * mask_seconddim
  dependency_seconddim = dataset.get_dependency(mask_seconddim, complete_x_seconddim, index=None, dataset_type = None)
  Y_second_dim = torch.sum(dependency_seconddim * aux_Y, axis=-1)
  
  
  # All dim needed :
  mask_all_dim = torch.ones_like(complete_X)
  dependency_all_dim = dataset.get_dependency(mask_all_dim, complete_X, index=None, dataset_type = None)
  Y_all_dim = torch.sum(dependency_all_dim * aux_Y, axis=-1)

  Y_first_dim_reshaped = Y_first_dim.reshape(1000,1000)
  Y_second_dim_reshaped = Y_second_dim.reshape(1000,1000)
  Y_all_dim_reshaped = Y_all_dim.reshape(1000,1000)



  
  true_selection_firstdim = torch.std(Y_first_dim_reshaped - Y_all_dim_reshaped, dim = -1, keepdim = True).expand(1000,1000)
  true_selection_seconddim = torch.std(Y_second_dim_reshaped - Y_all_dim_reshaped, dim = 0, keepdim=True).expand(1000,1000)
  
  max_std = torch.max(torch.cat([true_selection_firstdim, true_selection_seconddim], dim=0))

  true_selection_firstdim = 1 - true_selection_firstdim/max_std
  true_selection_seconddim = 1 - true_selection_seconddim/max_std




  aux = torch.cat([true_selection_firstdim.reshape(1,-1), true_selection_seconddim.reshape(1, -1)], dim=0) # Get true selection, 1 means selection; 0 means it's not necessary
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
  
  if train_data :
    for k in range(3):
      axs[k].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  fig.colorbar(
   ScalarMappable(norm=qcs.norm, cmap=qcs.cmap),
   ticks=range(0, 1, 20) 
  )


  complete_path = os.path.join(path, "true_dataset_selection.jpg")
  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)



def plot_destructor_output(destructor, dataset, path, train_data = False, nb_train_data = 1000, interpretation= False, figsize = (15,5),):
  try :
    centroids = dataset.centroids
  except :
    centroids = None


  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]



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
  pi_list = pi_list.clamp(min = 0.0, max=1.0)


  if interpretation :
    selection_bothdim, _ = torch.min(pi_list, dim = 1)
  else :
    selection_bothdim = torch.zeros_like(pi_list[:,0])


  # First dimension needed :
  selection_firstdim = pi_list[:,0] - selection_bothdim # Get true selection, 1 means selection; 0 means it's not necessary
  # Second dimension needed :
  selection_seconddim = pi_list[:,1] - selection_bothdim

  if interpretation :
    fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (15,5))
  else :
    fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (10,5))
  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                  int(max(Y) + 1))))

  axs[0].contourf(grid_x, grid_y, selection_firstdim.reshape(grid_x.shape),  vmin=0.0, vmax=1.0)
  axs[1].contourf(grid_x, grid_y, selection_seconddim.reshape(grid_x.shape), vmin=0.0, vmax=1.0)
  if interpretation :
    axs[2].contourf(grid_x, grid_y, selection_bothdim.reshape(grid_x.shape), vmin=0, vmax=1.0)
  if centroids is not None :
    for k in range(len(axs)):
      axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
  if train_data :
      for k in range(len(axs)):
        axs[k].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)


  complete_path = os.path.join(path, "selector_selection.jpg")

  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  if interpretation :
    complete_path = complete_path.split(".jpg")[0] + "_interpretation.jpg"


  plt.savefig(complete_path)
  plt.close(fig)




def plot_model_output(trainer_var, dataset, sampling_distribution, path, imputed_centroids = False, nb_train_data = 1000, train_data = False):
  try :
    centroids = dataset.centroids
  except :
    centroids = None
  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]


  if centroids is not None :
    if imputed_centroids :
      centroids_masks = dataset.new_S
      imputation = trainer_var.classification_module.imputation
      centroids, _ = imputation.impute(centroids, centroids_masks)    

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
    log_y_hat_destructed = trainer_var._predict(complete_X, sampling_distribution, 2, Nexpectation = 20, index = None)    
    pred_destruction = torch.exp(log_y_hat_destructed[:,0]).detach().cpu()
    destructive= True

  except(AttributeError) as e :
    print(e)
    destructive = False


  log_y_hat, _ = trainer_var.classification_module(complete_X, index = None)


  pred_classic = torch.exp(log_y_hat[:,0]).detach().cpu()



  # All dim needed :

  colors = np.array(list(islice(cycle([ '#ff7f00', '#377eb8', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                    int(max(Y) + 1))))
  if destructive :
    fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (10,5))
    axs[0].contourf(grid_x, grid_y, pred_classic.reshape(grid_x.shape),  vmin=0, vmax=1.0)
    axs[1].contourf(grid_x, grid_y, pred_destruction.reshape(grid_x.shape), vmin=0, vmax=1.0)
    if centroids is not None:
      for k in range(2):
        axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
    
    if train_data :
      for k in range(2):
        axs[k].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  else :
    fig, axs = plt.subplots(nrows =1, ncols = 1, figsize = (5,5))
    axs.contourf(grid_x, grid_y, pred_classic.reshape(grid_x.shape),  vmin=0, vmax=1.0)

    if centroids is not None:
        axs.scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
    if train_data :
        axs.scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  
  complete_path = os.path.join(path, "output_classification.jpg")


  if imputed_centroids :
    complete_path = complete_path.split(".jpg")[0] + "_imputed_centroids.jpg"
  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)



def get_dic_experiment(output_dic, to_save_dic,):
  for key in output_dic["test"].keys():
    save_key = key+ "_test"
    to_save_dic[save_key]= output_dic["test"][key][-1]
  return to_save_dic



def get_evaluation(trainer_var, loader, dic, dic_count_batch_size, dic_count_dim_batch_size, current_sampling_test, args_train, train = False):
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
            dic[key+suffix] = comparing_dic[key]

        for key in dic_count_batch_size.keys():
            dic_count_batch_size[key] = batch_size
        for key in dic_count_dim_batch_size.keys():
            dic_count_dim_batch_size[key] = dim * batch_size

    return dic, dic_count_batch_size, dic_count_dim_batch_size


def normalize_dic(dic, dic_count_batch_size, dic_count_dim_batch_size,):
  for key in dic_count_dim_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_dim_batch_size[key][k]
  for key in dic_count_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_batch_size[key][k]
  
  return dic