import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd

path_adult_dataset = r"D:\DTU\firstProject\MissingDataTraining\data\adultDataset\adult.csv"


default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])

default_Adult_transform = torchvision.transforms.Compose([
                                    ])

def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised


class FooDataset(Dataset):
    def __init__(self,shape = (3,3), len_dataset = 10000, shift = 3):
        self.size_x = shape[0]
        self.size_y = shape[1]
        self.nb_cat = self.size_x * self.size_y
        self.shift = shift 
        self.len = len_dataset
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ])
        

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        target = int(idx%(self.nb_cat-self.shift))
        x = np.ones((self.nb_cat), dtype=np.float32)
        x[(target+1):] = np.zeros((self.nb_cat - (target+1))) 
        x = x.reshape((self.size_x, self.size_y))

        data = self.transform(x)
        return data,target

class DatasetFoo():
    def __init__(self, batch_size_train, batch_size_test, shape = (3,3) , len_dataset = 10000):
        self.dataset = FooDataset(shape = shape, len_dataset = len_dataset)
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_train, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size_test, shuffle=True)

    def get_category(self):
        return self.dataset.nb_cat - self.dataset.shift

    def get_shape(self):
        return (1,self.dataset.size_x,self.dataset.size_y) 


## ======================= MNIST ======================================



# MNIST VARIATION :


class MnistDataset(torchvision.datasets.MNIST):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download)
        self.noisy = noisy
        self.noise_function = noise_function

       
    def __str__(self):
        return "SimpleMnist"
        
    def __getitem__(self, idx):
        if not self.noisy :
            img, target = self.data[idx], int(self.targets[idx])

            img = img.numpy()
            # target = target.numpy()
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        else :
            img, target = self.data[idx], self.data[idx]
            
            img = img.numpy()
            target = target.numpy()
      
            if self.transform is not None:
                target = self.transform(target)
                img = self.transform(img)

            img = self.noise_function(img).type(torch.float32)


            return img, target


class MnistVariation1(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.data_aux = []
        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        for element in self.data:
            index_new = np.random.randint(self.__len__())
            img_new = self.data[index_new]
            self.data_aux.append(element)
            self.data_aux[-1][middle_size_x:, middle_size_y:] = img_new[middle_size_x:, middle_size_y:]
        self.data = self.data_aux
           
    def __str__(self):
        return "MnistVariation1"
        

class MnistVariation2(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.data_aux = copy.deepcopy(self.data)
        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        for k in range(len(self.data)):
            index_new = np.random.randint(self.__len__())
            img_next, target_next = self.data[index_new], int(self.targets[index_new])
            if target_next > target :
                self.targets[k] = target_next
            self.data_aux[k][middle_size_x:, :] = img_next[middle_size_x:, :]

        self.data = self.data_aux
    def __str__(self):
        return "MnistVariation2"



class MnistVariationFashion(MnistDataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            noisy: bool = False,
            noise_function = None,
    ) :
        super().__init__(root, train, transform, target_transform, download, noisy, noise_function)
        self.fashion_mnist = torchvision.datasets.FashionMNIST(root, train, transform, target_transform, download)
        self.data_aux = copy.deepcopy(self.data)
        self.fashion_data = self.fashion_mnist.data

        middle_size_x, middle_size_y = int(np.shape(self.data[0])[-2]/2),int(np.shape(self.data[0])[-1]/2) 
        quadrant_x = np.random.randint(2, size = (len(self.data)))
        quadrant_y = np.random.randint(2, size = (len(self.data)))
        anchor_x_1 = middle_size_x *quadrant_x
        anchor_x_2 = anchor_x_1 + middle_size_x
        anchor_y_1 = middle_size_y * quadrant_y
        anchor_y_2 = anchor_y_1 + middle_size_y
        for k in range(len(self.data)):
            index_new = np.random.randint(len(self.fashion_data))
            img_next = self.fashion_data[index_new]
            self.data_aux[k][anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]] = img_next[anchor_x_1[k]: anchor_x_2[k], anchor_y_1[k] : anchor_y_2[k]]

        self.data = self.data_aux
    def __str__(self):
        return "MnistVariation2"

# MIX MNIST OMNIGLOT

##### ENCAPSULATION :

class LoaderEncapsulation():
    def __init__(self, dataset_class = MnistDataset, batch_size_train = 64, batch_size_test=1000, transform = default_MNIST_transform, noisy = False, noise_function = None ):
        self.dataset_class = dataset_class
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
     
        self.train_loader = torch.utils.data.DataLoader( dataset_class('/files/', train=True, download=True,
                                transform=transform, noisy=noisy, noise_function=noise_function),
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                dataset_class('/files/', train=False, download=True,
                                    transform=transform, noisy=noisy, noise_function=noise_function
                                                            ),
                            batch_size=batch_size_test, shuffle=True
                            )

    def get_category(self):
        return 10

    def get_shape(self):
        return (1,28,28)


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


##====================== SENTIMENT ANALYSIS ==============================
# from torchtext import data
# from torchtext import dataset
# class SentimentEncapsulation(Dataset):

#     def __init__(self, batch_size=64):
#         torch.backends.cudnn.deterministic = True
#         SEED = 1234

#         torch.manual_seed(SEED)
#         torch.backends.cudnn.deterministic = True
#         self.TEXT = data.Field(tokenize = 'spacy')
#         self.LABEL = data.LabelField(dtype = torch.float)

#         self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

        
#         self.MAX_VOCAB_SIZE = 25_000

#         self.TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
#         self.LABEL.build_vocab(train_data)
#         self.train_laoder, self.test_loader = data.BucketIterator.splits(
#             (self.train_data, self.test_data), 
#             batch_size = batch_size,
#             device = device
#         )

#     def get_category(self):
#         return 2

#     # def get_shape(s

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