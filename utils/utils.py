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

def show_interpretation_tabular(sample, data, target, dataloader):

 
    for i in range(len(sample)):
        list_element = []
        x_list = []
        print(f"Wanted target category : {target[i]}")
        for index in range(np.shape(data)[1]):
            print("Category : {} \t Proba : {} \t Value : {}".format(
                dataloader.train_loader.dataset.get_coded_category(index),
                sample[i][index],
                data[i][index]
            ))
            list_element.append(dataloader.train_loader.dataset.get_coded_category(index))
            x_list.append(index)
        x_list = np.arange(len(list_element))
        width = 0.35
        fig, ax = plt.subplots()
        ax1 = ax.bar(x_list-width/2, sample[i], width, label = 'Sample')
        ax2 = ax.bar(x_list+width/2, data[i], width, label = 'Data')
        ax.set_xticks(x_list)
        ax.set_xticklabels(list_element, rotation = 45, ha="right", rotation_mode = "anchor")
        ax.legend()
        plt.show()



def save_interpretation(path, sample, data, target, shape = (1,28,28),suffix = "", prefix= 'sample', y_hat = None, class_names = None):
  if not os.path.exists(path):
      os.makedirs(path)
  channels = shape[0]
  if y_hat is None :
      show_pred = False
      subplot_number = 2

  else :
      show_pred = True
      subplot_number = 3
      if class_names is None :
        class_names = [str(i) for i in range(len(y_hat[0]))]

 
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure()
        plt.subplot(1,subplot_number,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,subplot_number,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        if show_pred :
            plt.subplot(1, subplot_number, 3)
            x_pos = np.arange(0, len(y_hat[i]))
            plt.bar(x_pos,y_hat[i])
            plt.xticks(ticks = x_pos, labels=class_names)

        plt.savefig(os.path.join(path,f"{prefix}_{i}_target_{target[i]}_{suffix}.jpg"))





def fill_dic(total_dic, dic):
    if len(total_dic.keys())==0:
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key]=dic[key]
            else :
                total_dic[key] = [dic[key]]
    else :
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key].extend(dic[key])
            else :
                total_dic[key].append(dic[key])


    return total_dic

def save_dic(path, dic):
    if not os.path.exists(path):
        os.makedirs(path)

    for key in dic.keys():
        table = dic[key]
        plt.figure(0)
        plt.plot(np.linspace(0,len(table)-1,len(table)),table)
        plt.savefig(os.path.join(path,str(key)+".jpg"))
        plt.clf()



def train_autoencoder(autoencoder, dataset, optim):

    autoencoder.train()
    once = True
    for batch_idx, (data, target) in enumerate(dataset.train_loader):
        autoencoder.zero_grad()
        data = data.cuda()

        target = target.cuda()
        output = autoencoder(data).reshape(data.shape)
        loss = torch.nn.functional.mse_loss(output, target)
 


        if batch_idx % 100 == 0:
            percent = 100. * batch_idx / len(dataset.train_loader)
            
            print("[{}/{} ({:.0f}%)]\t {:.5f}".format(batch_idx*dataset.batch_size_train,
             len(dataset.train_loader.dataset), percent,loss.item()))
        loss.backward()
        optim.step()



def test_autoencoder(autoencoder, dataset):
    autoencoder.eval()

    loss = 0
    for batch_idx , (data, target) in enumerate(dataset.test_loader):
        data = data.cuda()
        target = target.cuda()
        output = autoencoder(data).reshape(data.shape)
        loss += torch.nn.functional.mse_loss(output, target, reduction = 'sum')

    loss /= len(dataset.test_loader.dataset) * dataset.batch_size_test
    print('\nTest set: AMSE: {:.4f}'.format(loss))


    


class TextPipeline():
    def __init__(self, list_classifier, list_prepare = []):
        self.list_classifier = list_classifier
        self.list_prepare = list_prepare

    def predict(self, data):
        for process in self.list_prepare :
            # print(data)
            data = process(data)

        output = data
        for process in self.list_classifier :
            output = process(output)

        output = output.detach().numpy()
        return output