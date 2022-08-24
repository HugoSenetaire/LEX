import sys
sys.path.append("C:\\Users\\hhjs\\Documents\\FirstProject\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining")

import traceback
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from itertools import cycle, islice

from missingDataTrainingModule.EvaluationUtils import eval_selection_sample, test_epoch, eval_selection

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])

### SAVINGS FOR HYPERCUDE DATASET

def plot_selector_output(selector, dataset, path, train_data = False, nb_train_data = 1000, interpretation= False, figsize = (15,5),):
  try :
    centroids = dataset.centroids
  except Exception as e:
    print(traceback.format_exc())
    centroids = None


  X = dataset.data_train
  Y = dataset.target_train

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  data_train = X[indexes, :]
  target_train = Y[indexes]

  nb_classes = dataset.get_dim_output()
  if nb_classes == 1 :
    classification = False
  else :
    classification = True




  min_x = torch.min(dataset.data_train[:,0])
  max_x = torch.max(dataset.data_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, 100)
  min_x = torch.min(dataset.data_train[:,1])
  max_x = torch.max(dataset.data_train[:,1])
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
      if classification :
        axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
      else :
        axs[k].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[0])
  if train_data :
      for k in range(len(axs)):
        if classification :
          axs[k].scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), color = colors[target_train.detach().cpu()], alpha = 0.3)
        else :
          axs[k].scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), color = "black", alpha = 0.3)

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
  except Exception as e:
    print(traceback.format_exc())
    centroids = None
  X = dataset.data_train
  Y = dataset.target_train
  nb_classes = dataset.get_dim_output()
  if nb_classes == 1 :
    classification = False
  elif nb_classes>1 :
    classification = True
  else :
    raise ValueError("Number of classes is not correct")

  indexes = np.random.choice(np.arange(0, len(Y)), min(nb_train_data, len(Y)), replace=False)
  data_train = X[indexes, :]
  target_train = Y[indexes]

  if centroids is not None :
    if imputed_centroids :
      centroids_masks = dataset.optimal_S
      imputation = trainer.prediction_module.imputation
      centroids, _ = imputation.impute(centroids, centroids_masks)    

  min_x = torch.min(dataset.data_train[:,0])
  max_x = torch.max(dataset.data_train[:,0])
  linspace_firstdim = torch.linspace(min_x, max_x, 100)
  min_x = torch.min(dataset.data_train[:,1])
  max_x = torch.max(dataset.data_train[:,1])
  linspace_seconddim = torch.linspace(min_x, max_x, 100)

  grid_x, grid_y = torch.meshgrid(linspace_firstdim, linspace_seconddim)
  xaux1 = grid_x.reshape(1, -1)
  xaux2 = grid_y.reshape(1, -1)
  complete_X = torch.cat([xaux1, xaux2], dim=0).transpose(1,0)

  
  if next(trainer.prediction_module.classifier.parameters()).is_cuda:
    complete_X = complete_X.cuda()

  if hasattr(trainer, "selection_module"):
    trainer.eval()
    log_pi_list, _ = trainer.selection_module(complete_X)
    trainer.distribution_module(torch.exp(log_pi_list))
    z = trainer.distribution_module.sample((1,))
    z = trainer.reshape(z)

    output_selection, _ = trainer.prediction_module(complete_X, z)
    # output_selection = trainer._predict(complete_X, sampling_distribution, nb_classes, Nexpectation = 20, index = None)  
    if classification :
      pred_selection = torch.exp(output_selection).detach().cpu()
    output_selection = output_selection.detach().cpu()
    destructive= True
  else :
    destructive = False
  


  output, _ = trainer.prediction_module(complete_X, index = None)
  if classification :
    pred_selection = torch.exp(output).detach().cpu()
  
  output = output.detach().cpu()

  # All dim needed :

  colors = np.array(list(islice(cycle([ '#ff7f00', '#377eb8', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                    int(max(Y) + 1))))
  if classification :
    if destructive :
      fig, axs = plt.subplots(nrows = nb_classes, ncols = 2, figsize = (10, nb_classes*5))
      for category_index in range(nb_classes):
        axs[category_index, 0,].contourf(grid_x, grid_y, pred_selection[:,category_index].reshape(grid_x.shape),  vmin=0, vmax=1.0)
        axs[category_index, 1,].contourf(grid_x, grid_y, pred_selection[:,category_index].reshape(grid_x.shape), vmin=0, vmax=1.0)
        if centroids is not None:
          for k in range(2):
            axs[category_index, k,].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
        
        if train_data :
          for k in range(2):
            axs[category_index, k,].scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), color = colors[target_train.detach().cpu()], alpha = 0.3)

    else :
      fig, axs = plt.subplots(nrows =nb_classes, ncols = 1, figsize = (5,nb_classes*5))
      for category_index in range(nb_classes):
        axs[category_index].contourf(grid_x, grid_y, pred_selection[:,category_index].reshape(grid_x.shape),  vmin=0, vmax=1.0)

        if centroids is not None:
            axs[category_index].scatter(centroids[:,0].detach().cpu(), centroids[:,1].detach().cpu(), color = colors[dataset.centroids_Y.detach().cpu()])
        if train_data :
            axs[category_index].scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), color = colors[target_train.detach().cpu()], alpha = 0.3)

      

  else :
    target_train = target_train.detach().cpu().numpy()
    if destructive :
      fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
      axs[0].contourf(grid_x, grid_y, output.reshape(grid_x.shape), vmin = np.min(output), vmax = np.max(output))
      axs[1].contourf(grid_x, grid_y, output_selection.reshape(grid_x.shape), vmin = np.min(output_selection), vmax = np.max(output_selection))
      if train_data :
        for k in range(2):
          axs[k].scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), c = target_train, vmin= np.min(output), vmax = np.max(output), cmap="viridis", alpha = 0.3) 
    else :
      fig = plt.figure(figsize = (5,5))
      plt.contourf(grid_x, grid_y, output.reshape(grid_x.shape),)
      if train_data :
        plt.scatter(data_train[:,0].detach().cpu(), data_train[:,1].detach().cpu(), c = target_train,  vmin= np.min(target_train), vmax = np.max(target_train), cmap="viridis", alpha = 0.3)
      
  complete_path = os.path.join(path, "output_classification.jpg")
  if imputed_centroids :
    complete_path = complete_path.split(".jpg")[0] + "_imputed_centroids.jpg"
  if train_data :
    complete_path = complete_path.split(".jpg")[0] + "_train_data.jpg"

  plt.savefig(complete_path)
  plt.close(fig)



##  Accuracy of the experiments :

def calculate_score(interpretable_module, loader, trainer, args, CFindex = None):
  interpretable_module.eval()
  dic = eval_selection_sample(interpretable_module, loader,)

  if CFindex is not None and f"Mean_pi_{CFindex}" in list(dic.keys()):
    dic["CPFR_rate"] = dic["Mean_pi_{}".format(CFindex)]

  for key in list(dic.keys()) :
    dic["sampled_" +key] = dic[key]

  dic.update(eval_selection(interpretable_module, loader, args))
  dic.update(test_epoch(interpretable_module, None, loader, args, liste_mc = [(1,1,1,1),], trainer = trainer,))

  return dic