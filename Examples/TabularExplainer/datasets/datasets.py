import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd

path_adult_dataset = r"D:\DTU\firstProject\MissingDataTraining\Examples\TabularExplainer\data\adultDataset\adult.csv"



default_Adult_transform = torchvision.transforms.Compose([
                                    ])

def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised

# UCI Adult dataset :
def prepare_adult_dataset(file_path = path_adult_dataset, train=False):
    pandas_dataset = pd.read_csv(file_path)
    # if train :
    #     pandas_dataset = pandas_dataset[:int(4*len(pandas_dataset)/5)]
    # else :
    #     pandas_dataset = pandas_dataset[int(4*len(pandas_dataset)/5):]


    attrib, counts = np.unique(pandas_dataset['workclass'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    pandas_dataset['workclass'][pandas_dataset['workclass'] == '?'] = most_freq_attrib 

    attrib, counts = np.unique(pandas_dataset['occupation'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    pandas_dataset['occupation'][pandas_dataset['occupation'] == '?'] = most_freq_attrib 

    attrib, counts = np.unique(pandas_dataset['native-country'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    pandas_dataset['native-country'][pandas_dataset['native-country'] == '?'] = most_freq_attrib 

    
    columns_input = ["age", "fnlwgt", "workclass", "education", "marital-status","occupation",
    "relationship", "race", "gender", "capital-gain", "capital-loss",
    "hours-per-week", "native-country"]



    
    
    pandas_dataset = pandas_dataset.sample(frac=1).reset_index(drop=True)

    df_input = pandas_dataset[columns_input]
    df_input["capital-diff"] = df_input["capital-gain"] - df_input["capital-loss"]
    df_input = df_input.drop(columns= ["capital-gain", "capital-loss"])

    cols_to_norm = df_input.drop(columns = df_input.select_dtypes(include=['object']).keys()).keys()
    df_input[cols_to_norm] = df_input[cols_to_norm].apply(lambda x: (x ) / (x.max() - x.min()))


    columns_output = ["income"]
    df_output = pandas_dataset[columns_output]


    key_to_encode = df_input.select_dtypes(include=['object']).keys()
    for key in key_to_encode :
      nb_values = len(df_input[key].unique())
      if nb_values == 2:
        df_input[key] = df_input[key].astype('category')
        df_input[f"{key}_encoded"] = df_input[str(key)].cat.codes
      else :
        df_input = pd.get_dummies(df_input, columns = [key], prefix= [key])
        list_key_to_delete = []
        other_key = []
        for test_key in df_input.keys():
            if not test_key.startswith(key) or test_key == key:
                other_key.append(test_key)
            else :
                if len(df_input[df_input[test_key]== 1])<len(df_input)/20 :
                    list_key_to_delete.append(test_key)
                else :
                    other_key.append(test_key)
        df_input[f"{key}_others"] = df_input.drop(columns = other_key).sum(axis=1, skipna = True)
        # print("===================")
        # print(key)
        # print(other_key)
        # print(list_key_to_delete)
        # print(df_input.drop(columns = other_key).keys())
        # print(df_input[f"{key}_others"].unique())
        df_input = df_input.drop(columns = list_key_to_delete)

    key_to_encode = df_output.select_dtypes(include=['object']).keys()
    for key in key_to_encode :
      df_output[key] = df_output[key].astype('category')
      df_output[f"{key}_encoded"] = df_output[str(key)].cat.codes

    return df_input, df_output



class AdultDataset(Dataset):
    def __init__(self, df_input, df_output, transform = default_Adult_transform, target_transform = None, noisy = False, noise_function = None):
        self.df_input = df_input
        self.df_output = df_output
        self.noisy = noisy
        self.noise_function = noise_function
        self.data = self.df_input.drop(columns = self.df_input.select_dtypes(include=['category']).keys()).to_numpy()
        self.index_to_key_data = self.df_input.drop(columns = self.df_input.select_dtypes(include=['category']).keys()).keys()

        self.targets = self.df_output.drop(columns = self.df_output.select_dtypes(include=['category']).keys()).to_numpy()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.noisy :
            x, target = self.data[idx], int(self.targets[idx])

            # x = x.numpy()
            # target = target.numpy()
            if self.transform is not None:
                x = self.transform(x)

            x = torch.tensor(x).type(torch.float32)

            if self.target_transform is not None:
                target = self.target_transform(target)
            target = torch.tensor(target)

            return x, target
        else :
            x, target = self.data[idx], self.data[idx]
            
            x = x
            target = target
      
            if self.transform is not None:
                target = self.transform(target)
                x = self.transform(x)
            x = torch.tensor(x).type(torch.float32)
            target = torch.tensor(target).type(torch.float32)
            x = self.noise_function(x).type(torch.float32)

            return x, target

        
    def get_coded_category(self, idx_key):
        return self.index_to_key_data[idx_key]
        
    def get_value(self, idx_key, value):
        coded_category =  self.index_to_key_data(idx_key)
        category = self.index_to_key_data(idx_key).strip("_encoded")

        new_df = self.df_input[self.df_input[idx_key]==value]
        return new_df[category][0]
        # raise NotImplementedError
    
class AdultDatasetEncapsulation():

    def __init__(self, dataset_class = AdultDataset, batch_size_train = 64, batch_size_test=1000, file_path = path_adult_dataset, transform = default_Adult_transform, noisy = False, noise_function = None ):
        self.dataset_class = dataset_class
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
        self.file_path = file_path
        self.df_input, self.df_output = prepare_adult_dataset(file_path=self.file_path)

        self.df_train_input = self.df_input[:int(4*len(self.df_input)/5)]
        self.df_test_input = self.df_input[int(4*len(self.df_input)/5):]
        self.df_train_output = self.df_output[:int(4*len(self.df_output)/5)]
        self.df_test_output = self.df_output[int(4*len(self.df_output)/5):]

     
        self.train_loader = torch.utils.data.DataLoader(dataset_class(self.df_train_input, self.df_train_output,
                                transform=transform, noisy=noisy, noise_function=noise_function),
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                dataset_class(self.df_test_input, self.df_test_output,
                                    transform=transform, noisy=noisy, noise_function=noise_function
                                                            ),
                            batch_size=batch_size_test, shuffle=True
                            )

    def get_category(self):
        return 2

    def get_shape(self):
        return (np.shape(self.train_loader.dataset.data)[1],)
