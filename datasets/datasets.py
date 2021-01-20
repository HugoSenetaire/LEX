import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import copy
import matplotlib.pyplot as plt


default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])


def create_noisy(data, noise_function):
    data_noised = []
    for k in range(len(data)):
        data_noised.append(noise_function(data[k]))

    return data_noised


class FooDataset(Dataset):
    def __init__(self,shape = (3,3), len_dataset = 10000, shift = 3):
        # assert(shape is tuple)
        # assert(len(shape)==2)
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
            # img =torch.where(img>torch.max(target),
            #      2*target-img, img
            #     )

            # img =torch.where(img<torch.min(target),
            #      2* target -img, img
            #     )
            # print("original",target[0][0])
            # print("troubled",img[0][0])

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

        # if self.noisy :
        #     if noise_function is None:
        #         raise AssertionError("Need to give noise function")
        #     self.data_noised = create_noisy(self.data, self.noise_function)

    # def __getitem__(self,idx):
    #     img, target = self.data[idx], int(self.targets[idx])

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Imagegit co


    #     index_new = np.random.randint(self.__len__())
    #     img_next, _ = self.data[index_new], int(self.targets[index_new])



    #     middle_size_x, middle_size_y = int(np.shape(img)[-2]/2),int(np.shape(img)[-1]/2) 
    #     img[middle_size_x:, middle_size_y:] = img_next[middle_size_x:, middle_size_y:]

    #     img_next = Image.fromarray(img_next.numpy(), mode='L')
    #     img = Image.fromarray(img.numpy(), mode='L')
    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target


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

        # self.noisy = noisy
        # self.noise_function = noise_function
        # if self.noisy :
        #     if noise_function is None:
        #         raise AssertionError("Need to give noise function")
        #     self.data_noised = create_noisy(self.data, self.noise_function)




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


