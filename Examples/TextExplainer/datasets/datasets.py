import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd


from sklearn.datasets import fetch_20newsgroups
import sklearn
import sklearn.ensemble
import sklearn.metrics
import scipy
class TextDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = torch.tensor(scipy.sparse.csr_matrix.todense(data)).float()
        self.targets = torch.tensor(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    


class ReligionDataset():
    def __init__(self, batch_size_train = 64, batch_size_test = 64):
        self.categories = ['alt.atheism', 'soc.religion.christian']
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test


        print("get_dataset")
        self.newsgroups_train = fetch_20newsgroups(subset='train', categories=self.categories)
        self.newsgroups_test = fetch_20newsgroups(subset='test', categories=self.categories)
        print("done")
        self.class_names = ['atheism', 'christian']

        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)

        self.train_vectors = self.vectorizer.fit_transform(self.newsgroups_train.data)
        print("vectorized")
        self.train_target = self.newsgroups_train.target
        self.dataset_train = TextDataset(self.train_vectors, self.train_target)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_vectors = self.vectorizer.transform(self.newsgroups_test.data)
        self.test_target = self.newsgroups_test.target
        self.dataset_test = TextDataset(self.test_vectors, self.test_target)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test,
                    batch_size=batch_size_test, shuffle=True
                    )


    def get_category(self):
        return 2

    def get_shape(self):
        return (np.shape(self.train_loader.dataset.data)[1],)