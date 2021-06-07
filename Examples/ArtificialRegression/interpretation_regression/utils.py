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