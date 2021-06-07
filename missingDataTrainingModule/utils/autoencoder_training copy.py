
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

