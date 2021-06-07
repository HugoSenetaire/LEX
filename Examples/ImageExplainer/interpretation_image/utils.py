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

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray, label2rgb

def show_interpretation(sample, data, target, shape = (1,28,28)):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,3,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.subplot(1,3,3)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=sample_reshaped[k].min().item(), vmax=sample_reshaped[k].max().item())
        plt.show()


def save_interpretation(path, sample, data, target, shape_sample = (1,28,28), shape_data = (1,28,28), suffix = "", prefix= 'sample', y_hat = None, class_names = None):
  if not os.path.exists(path):
      os.makedirs(path)

  if shape_sample[0] == 1:
      cmap_sample = 'gray'
  else :
      cmap_sample = None

  if shape_data[0] == 1:
      cmap_data = 'gray'
  else :
      cmap_data = None

  sample = np.transpose(sample,(0,2,3,1))
  data = np.transpose(data,(0,2,3,1))

  if y_hat is None :
      show_pred = False
      subplot_number = 2

  else :
      show_pred = True
      subplot_number = 3
      if class_names is None :
        class_names = [str(i) for i in range(len(y_hat[0]))]

 
  for i in range(len(sample)):
    path_target = os.path.join(path, f"target_{target[i]}")
    if not os.path.exists(path_target):
      os.makedirs(path_target)
    if prefix == "sample" :
      path_sample = os.path.join(path_target,f"sample_{i}")
      if not os.path.exists(path_sample):
        os.makedirs(path_sample)
    else :
      path_sample = path_target
    print(f"Wanted target category : {target[i]}")
    # sample_reshaped = sample[i].reshape(shape_sample)


    vmin_sample = np.min(sample)
    vmax_sample = np.max(sample)
    
    fig = plt.figure()
    plt.subplot(1,subplot_number,1)
    plt.imshow(data[i], cmap=cmap_data, interpolation='none')
    plt.subplot(1,subplot_number,2)
    plt.imshow(sample[i], cmap=cmap_sample, interpolation='none', vmin=vmin_sample, vmax=vmax_sample)
    if show_pred :
        plt.subplot(1, subplot_number, 3)
        x_pos = np.arange(0, len(y_hat[i]))
        plt.bar(x_pos,y_hat[i])
        plt.xticks(ticks = x_pos, labels=class_names)

    plt.savefig(os.path.join(path_sample,f"{prefix}_{i}_target_{target[i]}_{suffix}.jpg"))
    plt.close(fig)


def batch_predict_gray(images, model, feature_extractor = None):
    model.eval()
    if len(np.shape(images))== 3 :
      batch = torch.stack(tuple(torch.tensor(image, dtype= torch.float32) for image in images),dim = 0).unsqueeze(1)
    else :
      batch = torch.mean(torch.stack(tuple(torch.tensor(image, dtype= torch.float32) for image in images),dim = 0),dim=-1)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    if feature_extractor is not None :
      feature_extractor.to(device)
      batch = feature_extractor(batch)
    probs = model(batch)
    probs = torch.exp(probs)
    return probs.detach().cpu().numpy()


def batch_predict_gray_with_destruction(images, model_function):
    if len(np.shape(images))== 3 :
      batch = torch.stack(tuple(torch.tensor(image, dtype= torch.float32) for image in images),dim = 0).unsqueeze(1)
    else :
      batch = torch.mean(torch.stack(tuple(torch.tensor(image, dtype= torch.float32) for image in images),dim = 0),dim=-1).unsqueeze(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    probs = model_function(batch)
    probs = torch.exp(probs)
    return probs.detach().cpu().numpy()

defaultSegmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

def lime_image_interpretation(file_path, folder_name, data, targets, pipeline_predict, segmenter = defaultSegmenter, top_labels = 10, hide_color = 0, num_samples = 10000, class_names = None):
  explainer = lime_image.LimeImageExplainer()


  for i, (image, target) in enumerate(zip(data,targets)) :
      path_current = os.path.join(file_path,f"sample_{i}")
      if not os.path.exists(path_current):
        os.makedirs(path_current)

      path_current = os.path.join(path_current, folder_name)
      if not os.path.exists(path_current):
        os.makedirs(path_current)

      aux_image = image.squeeze().numpy().astype(np.double)
      score = pipeline_predict([aux_image])
      if len(np.shape(score))>=2:
        score = score[0]
      fig = plt.figure()
      x_pos = np.arange(0, len(score))
      plt.bar(x_pos,score)
      if class_names is None :
        class_names = np.arange(0, len(score))

      plt.xticks(ticks = x_pos, labels=class_names)
      plt.savefig(os.path.join(path_current,"score_lime.jpg"))
      plt.close(fig)

      explanation = explainer.explain_instance(aux_image, 
                                         classifier_fn = pipeline_predict, 
                                         top_labels= top_labels, hide_color= hide_color, num_samples= num_samples, segmentation_fn=segmenter)
        

      for i in range(top_labels):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, negative_only = False, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4))
        ax1.imshow(mask, interpolation = 'nearest')
        ax1.set_title('Positive Regions for {}'.format(i))
        temp, mask = explanation.get_image_and_mask(i, positive_only=False, negative_only = True, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(mask, interpolation = 'nearest')
        ax2.set_title('Negative Regions for {}'.format(i))

        ax3.imshow(temp)
        ax3.set_title('Original image')
        plt.savefig(os.path.join(path_current, f"Estimated_region_{i}.jpg"))
        plt.close(fig)



    