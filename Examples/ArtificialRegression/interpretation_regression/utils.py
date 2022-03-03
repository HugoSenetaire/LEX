import sys

from numpy.core.function_base import linspace

from missingDataTrainingModule.Classification.classification_network import ClassifierLVL3, ClassifierLinear
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
sys.path.append("/home/hhjs/MissingDataTraining")

import torch
import matplotlib.pyplot as plt

from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
from itertools import cycle, islice
import sklearn

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

list_name_post_hoc = [
  "PostHoc_NoArgmax_NoAuxiliary",
  "PostHoc_Argmax_NoAuxiliary",
  "PostHoc_Argmax_ClassifierLinear",
  "PostHoc_Argmax_ClassifierLVL3",
  "PostHoc_NoArgmax_ClassifierLinear",
  "PostHoc_NoArgmax_ClassifierLVL3",
  ]


def get_post_hoc_parameters(name, args_train):
  _, argmax_value, auxiliary_value = name.split("_")

  args_train["post_hoc"] = True
  if argmax_value == "Argmax":
    args_train["argmax_post_hoc_classification"] = True
  elif argmax_value == "NoArgmax":
    args_train["argmax_post_hoc_classification"] = False
  else :
    raise ValueError("argmax_value should be 'Argmax' or 'NoArgmax'")
    
  if auxiliary_value == "NoAuxiliary":
    args_train["fix_classifier_parameters"] = True
    args_train["nb_epoch_pretrain"] = 10
    args_train["post_hoc_guidance"] = None
  else :
    args_train["fix_classifier_parameters"] = False
    args_train["nb_epoch_post_hoc"] = 10
    args_train["nb_epoch_pretrain"] = 0
    if auxiliary_value == "ClassifierLinear":
      args_train["post_hoc_guidance"] = ClassifierLinear
    elif auxiliary_value == "ClassifierLVL3":
      args_train["post_hoc_guidance"] = ClassifierLVL3
    else:
      raise ValueError("Unknown post hoc guidance")


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
 "proportion_pi", "proportion_z", "proportion_thresholded_pi", "proportion_thresholded_z", "accuracy_prediction_no_selection",
  "accuracy_prediction_selection", "mean_pi_list", "pi_list_q1", "pi_list_q2", "pi_list_median","auc_score_pi", "auc_score_z"]

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


def plot_true_interpretation_v3(dataset, path, train_data = False, nb_train_data = 1000, grid_sample = 100, nb_imputation = 1000,  figsize=(15,5), interpretation = False):
  if dataset.nb_dim > 2 :
    return None
  
  try :
    centroids = dataset.centroids
  except AttributeError:
    centroids = None

  X = dataset.X_train
  Y = dataset.Y_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]



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
  complete_X = complete_X.type(torch.float64)



  true_selection = dataset.calculate_true_selection_variation(complete_X, classifier = None, nb_imputation=nb_imputation, normalize = True)


  true_selection_firstdim = true_selection[:,0]
  true_selection_seconddim = true_selection[:,1]
  aux = torch.cat([true_selection_firstdim.unsqueeze(0), true_selection_seconddim.unsqueeze(0)], dim=0)
  true_selection_alldim = torch.min(aux, dim=0)[0]
  

  if interpretation :
    true_selection_firstdim = true_selection_firstdim - true_selection_alldim
    true_selection_seconddim =  true_selection_seconddim - true_selection_alldim

  fig, axs = plt.subplots(nrows =1, ncols = 3, figsize = (15,5))
  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                  2)))

  axs[0].contourf(grid_x, grid_y, true_selection_firstdim.reshape(grid_x.shape),  vmin=0, vmax=1.0)
  axs[1].contourf(grid_x, grid_y, true_selection_seconddim.reshape(grid_x.shape), vmin=0, vmax=1.0)
  qcs = axs[2].contourf(grid_x, grid_y, true_selection_alldim.reshape(grid_x.shape), vmin=0, vmax=1.0)

  if centroids is not None:
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
  print("Figure saved at", complete_path)
  if interpretation :
    complete_path = complete_path.split(".jpg")[0] + "_interpretation.jpg"
  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)


def plot_selector_output(selector, dataset, path, train_data = False, nb_train_data = 1000, interpretation= False, figsize = (15,5),):
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

  if next(selector.selector.parameters()).is_cuda:
    complete_X = complete_X.cuda()
  log_pi_list, _ = selector(complete_X)
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


def plot_complete_model_output(trainer, dataset, sampling_distribution, path, imputed_centroids = False, nb_train_data = 1000, train_data = False):
  try :
    centroids = dataset.centroids
  except :
    centroids = None
  X = dataset.X_train
  Y = dataset.Y_train
  nb_classes = dataset.nb_classes

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  X_train = X[indexes, :]
  Y_train = Y[indexes]


  if centroids is not None :
    if imputed_centroids :
      centroids_masks = dataset.optimal_S
      imputation = trainer.classification_module.imputation
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

  
  if next(trainer.classification_module.classifier.parameters()).is_cuda:
    complete_X = complete_X.cuda()

  if hasattr(trainer, "selection_module"):
    trainer.eval()
    log_pi_list, _ = trainer.selection_module(complete_X)
    trainer.distribution_module(torch.exp(log_pi_list))
    z = trainer.distribution_module.sample((1,))
    z = trainer.reshape(z)

    log_y_hat_destructed, _ = trainer.classification_module(complete_X, z)
    # log_y_hat_destructed = trainer._predict(complete_X, sampling_distribution, nb_classes, Nexpectation = 20, index = None)  
    pred_selection = torch.exp(log_y_hat_destructed).detach().cpu()
    destructive= True
  else :
    destructive = False
  


  log_y_hat, _ = trainer.classification_module(complete_X, index = None)
  pred_classic = torch.exp(log_y_hat).detach().cpu()
  


  # All dim needed :

  colors = np.array(list(islice(cycle([ '#ff7f00', '#377eb8', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                    int(max(Y) + 1))))
  if destructive :
    fig, axs = plt.subplots(nrows = nb_classes, ncols = 2, figsize = (10, nb_classes*5))
    for category_index in range(nb_classes):
      axs[category_index, 0,].contourf(grid_x, grid_y, pred_classic[:,category_index].reshape(grid_x.shape),  vmin=0, vmax=1.0)
      axs[category_index, 1,].contourf(grid_x, grid_y, pred_selection[:,category_index].reshape(grid_x.shape), vmin=0, vmax=1.0)
      if centroids is not None:
        for k in range(2):
          axs[category_index, k,].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
      
      if train_data :
        for k in range(2):
          axs[category_index, k,].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

  else :
    fig, axs = plt.subplots(nrows =1, ncols = 1, figsize = (5,nb_classes*5))
    for category_index in range(nb_classes):
      axs[nb_classes].contourf(grid_x, grid_y, pred_classic[:,category_index].reshape(grid_x.shape),  vmin=0, vmax=1.0)

      if centroids is not None:
          axs[category_index].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
      if train_data :
          axs[category_index].scatter(X_train[:,0].detach().cpu(), X_train[:,1].detach().cpu(), color = colors[Y_train.detach().cpu()], alpha = 0.3)

    
  complete_path = os.path.join(path, "output_classification.jpg")


  if imputed_centroids :
    complete_path = complete_path.split(".jpg")[0] + "_imputed_centroids.jpg"
  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)



##  Accuracy of the experiments :



def calculate_score(trainer, loader,):
  trainer.classification_module.imputation.nb_imputation_mc_test = 1
  trainer.classification_module.imputation.nb_imputation_iwae_test = 1     
  trainer.eval()
  

  X_test = loader.dataset.X_test.type(torch.float32)
  Y_test = loader.dataset.Y_test.type(torch.float32)
  optimal_S_test = loader.dataset.optimal_S_test.type(torch.float32)

  if hasattr(trainer, "selection_module"):
    selection_module = trainer.selection_module.cpu()
    distribution_module = trainer.distribution_module.cpu()
    selection_module.eval()
    distribution_module.eval()
    selection_evaluation = True
  else :
    selection_evaluation = False
  classification_module = trainer.classification_module.cpu()


  classification_module.eval()


  with torch.no_grad():
    if selection_evaluation :
      log_pi_list, _ = selection_module(X_test)
      distribution_module(torch.exp(log_pi_list))
      z = distribution_module.sample((1,))
      z = trainer.reshape(z)
      log_y_hat, _ = classification_module(X_test, z, index=None)
      pred_classic = torch.exp(log_y_hat).detach().cpu().numpy()
      log_pi_list = log_pi_list.detach().cpu().numpy()
      if selection_module.activation is torch.nn.Softmax():
        log_pi_list = torch.log(distribution_module.sample((1000,)).mean(dim = 0))
        # log_pi_list = log_pi_list * np.prod(log_pi_list.shape[1:])

    log_y_hat_true_selection, _ = classification_module(X_test, optimal_S_test, index = None, )
    pred_true_selection = torch.exp(log_y_hat_true_selection).detach().cpu().numpy()

    log_y_hat_no_selection, _ = classification_module(X_test, index = None)
    pred_no_selection = torch.exp(log_y_hat_no_selection).detach().numpy()


  Y_test = Y_test.detach().cpu().numpy()
  optimal_S_test = optimal_S_test.detach().cpu().numpy()

  dic = {}
  if selection_evaluation:
    dic["accuracy"] = 1 - np.mean(np.abs(np.argmax(pred_classic, axis=1) - Y_test))
    dic["auroc"] = sklearn.metrics.roc_auc_score(Y_test, pred_classic[:,1])

  dic["accuracy_true_selection"] = 1 - np.mean(np.abs(np.argmax(pred_true_selection, axis=1) - Y_test))
  dic["auroc_true_selection"] = sklearn.metrics.roc_auc_score(Y_test, pred_true_selection[:,1])
  dic["accuracy_no_selection"] = 1 - np.mean(np.abs(np.argmax(pred_no_selection, axis=1) - Y_test))
  dic["auroc_no_selection"] = sklearn.metrics.roc_auc_score(Y_test, pred_no_selection[:,1])


  if selection_evaluation:
    dic["CPFSelection"] = np.sum(np.exp(log_pi_list[:,10]) > 0.5)/len(log_pi_list)
    dic["CPFR_rate"] = np.mean(np.exp(log_pi_list[:,10]))

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(optimal_S_test.reshape(-1), np.exp(log_pi_list).reshape(-1),)
    
    dic["fpr"] = fpr
    dic["tpr"] = tpr
    dic["thresholds"] = thresholds

    sel_pred = (np.exp(log_pi_list) >0.5).astype(int).reshape(-1)
    sel_true = optimal_S_test.reshape(-1)
    fp = np.sum((sel_pred == 1) & (sel_true == 0))
    tp = np.sum((sel_pred == 1) & (sel_true == 1))

    fn = np.sum((sel_pred == 0) & (sel_true == 1))
    tn = np.sum((sel_pred == 0) & (sel_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)


    dic["fpr2"] = fpr
    dic["tpr2"] = tpr

    dic["selection_auroc"] = sklearn.metrics.roc_auc_score(optimal_S_test.reshape(-1), np.exp(log_pi_list).reshape(-1))
    dic["selection_accuracy"] = 1 - np.mean(np.abs(optimal_S_test.reshape(-1) - np.round(np.exp(log_pi_list.reshape(-1)))))
    dic["mean_selection"] = np.mean(np.exp(log_pi_list), axis=0)


  return dic