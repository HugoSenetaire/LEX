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

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
  ax1.scatter(data[:,0],data[:, 1], s=10, color = colors[target])
  ax1.set_title('Input Data')
  # print(data.shape)
  # print(predicted.shape)
  # print(predicted)
  # print(data)
  ax2.scatter(data[:,0],data[:, 1], s=10, color = colors[predicted])
  ax2.set_title('Prediction')
  plt.savefig(os.path.join(path_result, f"Prediction output.jpg"))
  plt.close(fig)

def save_interpretation_artificial(path, data_destructed, target, predicted, weights = None, bias = None):
  path_result = os.path.join(path, "result")
  if not os.path.exists(path_result):
    os.makedirs(path_result)

  # if weights is None or bias is None :
  #   need_line = False
  # else :
  #   need_line = True


  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
  ax1.scatter(data_destructed[:,0], data_destructed[:,1], s=10, color = colors[target])
  # if need_line :
  #   x0 = min(data_destructed[:,0])
  #   x1 = max(data_destructed[:,0])
  #   x_list = np.linspace(x0, x1)
  #   # print(weights[0,0])
  #   # print(bias)
  #   y_list = -weights[0,0]/weights[0,1] * x_list - bias[0]/weights[0,1]
  #   print(y_list)
  #   ax1.plot(x_list, y_list, '-r', color = "b")

  # if need_line :
  #   x0 = min(data_destructed[:,0])
  #   x1 = max(data_destructed[:,0])
  #   x_list = np.linspace(x0, x1)
  #   # print(weights[0,0])
  #   # print(bias)
  #   y_list = -weights[1,0]/weights[1,1] * x_list - bias[1]/weights[1,1]
  #   print(y_list)
  #   ax1.plot(x_list, y_list, '-r', color = "r")

  ax1.set_title('Destructed data with target')
  ax2.scatter(data_destructed[:,0], data_destructed[:,1], s=10, color = colors[predicted])
  ax2.set_title('Destructed data with prediction')
  plt.savefig(os.path.join(path_result, f"Destructed_output.jpg"))
  plt.close(fig)

def save_interpretation_artificial_bar(path, sample, target, pred):
  path_result = os.path.join(path, "result")
  if not os.path.exists(path_result):
    os.makedirs(path_result)

  sample_0_real_1 = sample[np.where(target==1),0][0]
  # print(np.shape(sample_0_real_1))
  sample_0_real_0 = sample[np.where(target==0),0][0]

  sample_1_real_1 = sample[np.where(target==1),1][0]
  sample_1_real_0 = sample[np.where(target==0),1][0]



  fig = plt.figure(1)
  plt.boxplot([sample_0_real_0, sample_0_real_1, sample_1_real_0, sample_1_real_1],
     labels = ["Coord 0 Y 0", "Coord 0 Y 1", "Coord 1 Y 0" , "Coord 1 Y 1"])
  
  plt.savefig(os.path.join(path_result, f"box_plot_real_target.jpg"))
  plt.close(fig)


  sample_0_real_1 = sample[np.where(pred==1),0][0]
  sample_0_real_0 = sample[np.where(pred==0),0][0]

  sample_1_real_1 = sample[np.where(pred==1),1][0]
  sample_1_real_0 = sample[np.where(pred==0),1][0]



  fig = plt.figure(1)
  plt.boxplot([sample_0_real_0, sample_0_real_1, sample_1_real_0, sample_1_real_1],
     labels = ["Coord 0 y 0", "Coord 0 y 1", "Coord 1 y 0" , "Coord 1 y 1"])
  
  plt.savefig(os.path.join(path_result, f"box_plot_real_pred.jpg"))
  plt.close(fig)