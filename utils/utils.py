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
        fig = plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,3,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.subplot(1,3,3)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=sample_reshaped[k].min().item(), vmax=sample_reshaped[k].max().item())
        plt.show()

def save_interpretation(path, sample, data, target, shape = (1,28,28),suffix = "", prefix= 'sample'):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.savefig(os.path.join(path,f"{prefix}_{target[i]}_{suffix}.jpg"))


def fill_dic(total_dic, dic):
    if len(total_dic.keys())==0:
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key]=dic[key]
            else :
                total_dic[key] = [dic[key]]

    else :
        for key in dic.keys():
            # print(dic[key])
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


    

        