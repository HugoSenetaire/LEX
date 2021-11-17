import sys

from numpy.core.function_base import linspace

from missingDataTrainingModule.Classification.classification_network import ClassifierLVL3, ClassifierLinear
from missingDataTrainingModule.Destruction import destruction_module
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





def plot_true_interpretation_v3(dataset, path, train_data = False, nb_train_data = 1000, grid_sample = 100, nb_imputation = 1000,  figsize=(15,5), interpretation = False):
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




def plot_complete_model_output(trainer_var, dataset, sampling_distribution, path, imputed_centroids = False, nb_train_data = 1000, train_data = False):
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

  if hasattr(trainer_var, "destruction_module"):
    log_y_hat_destructed = trainer_var._predict(complete_X, sampling_distribution, nb_classes, Nexpectation = 20, index = None)  
    pred_destruction = torch.exp(log_y_hat_destructed).detach().cpu()
    destructive= True
  else :
    destructive = False
  


  log_y_hat, _ = trainer_var.classification_module(complete_X, index = None)
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
      axs[category_index, 1,].contourf(grid_x, grid_y, pred_destruction[:,category_index].reshape(grid_x.shape), vmin=0, vmax=1.0)
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




   
def compare_selection_single(dic, mask, true_masks, normalized_output = True, threshold_destructor = 0.5, suffix = "pi", true_masks_int = False):
    batch_size = len(mask)
    nb_dim = mask.shape[1]

    for dim in range(nb_dim):
      
      current_mask = mask.detach().cpu()[:, dim]
      true_current_masks = true_masks.detach().cpu()[:, dim]
      current_mask_thresholded = torch.where(current_mask > threshold_destructor, torch.ones_like(current_mask), torch.zeros_like(current_mask))
      accuracy = torch.sum(1-torch.abs(true_current_masks - current_mask))
      accuracy_thresholded = torch.sum(1-torch.abs(true_current_masks - current_mask_thresholded))


      if true_masks_int :
        try :
            auc_score = sklearn.metrics.roc_auc_score(true_current_masks.flatten().detach().cpu().numpy(), current_mask.flatten().detach().cpu().numpy())
            auc_score = torch.tensor(auc_score).type(torch.float32)
        except :
            auc_score = torch.abs(true_current_masks - current_mask).mean()

        confusion_matrix = sklearn.metrics.confusion_matrix(true_current_masks.flatten().detach().cpu().numpy(), current_mask_thresholded.flatten().detach().cpu().numpy())
        if normalized_output : 
          dic[f"dim_{dim}_auc_score_"+suffix] = auc_score.item()
          dic[f"dim_{dim}_confusion_matrix_"+suffix] = confusion_matrix / batch_size
        else :
          dic[f"dim_{dim}_auc_score_"+suffix] = auc_score.item() * batch_size
          dic[f"dim_{dim}_confusion_matrix_"+suffix] = confusion_matrix 

      if normalized_output : 
          accuracy = accuracy/batch_size
          accuracy_thresholded = accuracy_thresholded/batch_size

      dic[f"dim_{dim}_accuracy_"+suffix] = accuracy.item()
      dic[f"dim_{dim}_accuracythresholded_"+suffix] = accuracy_thresholded.item()

    return dic

def compare_selection(mask, true_masks, normalized_output = False, train_dataset = False, sampling_distribution = None, threshold_destructor = 0.5, nb_sample_z = 100, true_masks_int = False): 
    dic = {}
    dic = compare_selection_single(dic, mask, true_masks, normalized_output = normalized_output, threshold_destructor = threshold_destructor, suffix = "pi")

    # if sampling_distribution is not None :
    #     mask_z = sampling_distribution(probs=mask).sample((nb_sample_z,)).flatten(0,1)
    #     true_masks_z = true_masks.unsqueeze(0).expand(torch.Size((nb_sample_z,)) + true_masks.shape).flatten(0,1)
    #     dic = compare_selection_single(dic, mask_z, true_masks_z, normalized_output = normalized_output, threshold_destructor = threshold_destructor, suffix = "z", true_masks_int = true_masks_int)
    
    return dic


def get_evaluation_adhoc(trainer_var, loader, dic, current_sampling_test, args_train, train = False, nb_sample_z = 100,):
  if train:
      suffix_train = "_train_adhoc"
      wanted_loader = loader.train_loader
      dataset = loader.dataset
      true_masks_nopostprocess = dataset.S_train_dataset_based_unnormalized
  else :
      suffix_train = "_test_adhoc"
      wanted_loader = loader.test_loader
      dataset = loader.dataset
      true_masks_nopostprocess = dataset.S_test_dataset_based_unnormalized

  for aux_method in ["max_normalized", "true_thresholded_05" ]:
    if aux_method == "max_normalized" :
      true_masks = true_masks_nopostprocess/torch.max(true_masks_nopostprocess)
      suffix = suffix_train + "_max_normalized"
      true_masks_int = False
    elif aux_method == "true_thresholded_05" :
      true_masks = torch.where(true_masks_nopostprocess > 0.5, torch.ones_like(true_masks_nopostprocess), torch.zeros_like(true_masks_nopostprocess))
      suffix = suffix_train + "_true_thresholded_05"
      true_masks_int = True
    else :
      raise ValueError("Unknown aux_method")

    complete_batch_size = 0
    for (data, target, index) in iter(wanted_loader):
        if args_train["use_cuda"] :
          data = data.cuda()
          index = index.cuda()
        pi_list, log_pi_list,  _, z_s, _ = trainer_var._destructive_test(data, current_sampling_test, 1)

        batch_size, dim = data.shape
        current_true_masks = true_masks[index]
        if args_train["use_cuda"]:
            data = data.cuda()

        comparing_dic = compare_selection(mask = pi_list, true_masks = current_true_masks, normalized_output=False, train_dataset = train, sampling_distribution = current_sampling_test, nb_sample_z=nb_sample_z, true_masks_int=true_masks_int)

        for key in comparing_dic.keys():
            if key+suffix in dic.keys():
                dic[key+suffix] += comparing_dic[key]
            else :
                dic[key+suffix] = comparing_dic[key]
        complete_batch_size += batch_size

  for key in dic.keys():
    dic[key] = dic[key]/complete_batch_size
    print(key, dic[key])

  print("========================================================")
  return dic

def get_evaluation_posthoc(trainer_var, loader, dic, current_sampling_test, args_train, train = False, nb_sample_z = 100,):
  if train:
      suffix_train = "_train_posthoc"
      wanted_loader = loader.train_loader
      dataset = loader.dataset
  else :
      suffix_train = "_test_posthoc"
      wanted_loader = loader.test_loader
      dataset = loader.dataset

  for aux_method in ["max_normalized", "true_thresholded_05" ]:
    if aux_method == "max_normalized" :
      suffix = suffix_train + "_max_normalized"
      true_masks_int = False
    elif aux_method == "true_thresholded_05" :
      suffix = suffix_train + "_true_thresholded_05"
      true_masks_int = True
    else :
      raise ValueError("Unknown aux_method")
    

    complete_batch_size = 0
    for (data, target, index) in iter(wanted_loader):
        pi_list, log_pi_list,  _, z_s, _ = trainer_var._destructive_test(data, current_sampling_test, 1)
        batch_size, dim = data.shape
        if args_train["use_cuda"] :
          data = data.cuda()
          index = index.cuda()

        
        true_masks_nopostprocess = dataset.calculate_true_selection_variation(data, classifier = trainer_var.classification_module.classifier)

        if aux_method == "max_normalized" :
          true_masks = true_masks_nopostprocess/torch.max(true_masks_nopostprocess)
        elif aux_method == "true_thresholded_05" :
          true_masks = torch.where(true_masks_nopostprocess > 0.5, torch.ones_like(true_masks_nopostprocess), torch.zeros_like(true_masks_nopostprocess))

              
        if args_train["use_cuda"]:
            data = data.cuda()

        comparing_dic = compare_selection(mask = pi_list, true_masks = true_masks, normalized_output=False, train_dataset = train, sampling_distribution = current_sampling_test, nb_sample_z=nb_sample_z, true_masks_int=true_masks_int)
        
        for key in comparing_dic.keys():
            if key+suffix in dic.keys():
                dic[key+suffix] += comparing_dic[key]
            else :
                dic[key+suffix] = comparing_dic[key]
        complete_batch_size += batch_size
  for key in dic.keys():
    dic[key] = dic[key]/complete_batch_size


  return dic


def normalize_dic(dic, dic_count_batch_size, dic_count_dim_batch_size,):
  for key in dic_count_dim_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_dim_batch_size[key][k]
  for key in dic_count_batch_size.keys() :
      for k in range(len(dic[key])):
          dic[key][k] /= dic_count_batch_size[key][k]
  
  return dic