import torch
import torchvision 
import numpy as np 

from torch.utils.data import Dataset, DataLoader

from .dataset_from_data import DatasetFromData
from skimage.draw import polygon, polygon2mask
import PIL
import os

default_CELEBA_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(128),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

import matplotlib.pyplot as plt
def get_mouth(point_left, point_right):
    """
    Create a ground truth corresponding to a rectangle with the same direction as the line between the two points and a height corresponding at 0.4 of the distance between the two points.
    """
    direction = point_right - point_left
    direction = direction.to(torch.float32)
    direction = direction / torch.linalg.norm(direction)
    orthogonal = torch.stack([-direction[0,1], direction[0,0]])
    height = 0.4 * np.linalg.norm(point_right - point_left)
    bb = torch.cat([point_left-(height/2)*orthogonal, point_left + (height/2) * orthogonal, point_right + (height/2) * orthogonal, point_right - height/2 * orthogonal]).to(torch.int64)
    mouth = polygon2mask((128,128),bb.numpy())
    return mouth



class EncapsulateCelebA(Dataset):
    def __init__(self, dataset, target_index):
        super().__init__()
        self.dataset = dataset
        self.target_index = target_index
    def __len__(self):
        return 100

    def __getitem__(self, index):
        image, target = self.dataset.__getitem__(index)
        true_target = target[0][self.target_index].to(torch.int64)
        return image, true_target, index
        

class CELEBA():
    def __init__(self,
            root_dir: str,
            transform = default_CELEBA_transform,
            target_transforms = None,
            download: bool = False,
            target = "Smiling",
            **kwargs,):

        self.celeba_train = torchvision.datasets.CelebA(root = root_dir, split="train", download=download, transform = transform, target_type=["attr", "landmarks"])
        self.celeba_test  = torchvision.datasets.CelebA(root = root_dir, split="test", download=download, transform = transform, target_type=["attr", "landmarks"])
        
        assert target in order_target
    
        self.target_index = order_target.index(target)


        self.dataset_train = EncapsulateCelebA(self.celeba_train, self.target_index)
        self.dataset_test = EncapsulateCelebA(self.celeba_test, self.target_index)

    def get_true_selection(self, indexes, type = "test",):
        """
        Return the true selection for the given indexes but without storing it in memory.
        """


        if type == "train" :
            dataset = self.celeba_train
        elif type == "test" :
            dataset = self.celeba_test
        else :
            raise ValueError("dataset_type must be either train or test")

        mouth_list = []
        for index in indexes :
            out_celeba = dataset.__getitem__(index)
            data = out_celeba[0]
            landmarks = out_celeba[1][1].reshape(1,10)
            mouth_landmarks_left= landmarks[:,[7,6]].flatten(1) - torch.tensor([45, 25])
            mouth_landmarks_right= landmarks[:,[9,8]].flatten(1) - torch.tensor([45, 25])
            mouth_list += [torch.tensor(get_mouth(mouth_landmarks_left, mouth_landmarks_right))]

        optimal_S = torch.stack(mouth_list).reshape(-1,1,128,128)
        return optimal_S



    def get_dim_input(self,):
        return (3,128,128)
        
    def get_dim_output(self,):
        return 2


    def __str__(self):
        return "Mnist_and_FashionMNIST"



order_target = ["5_o_Clock_Shadow",
"Arched_Eyebrows",
"Attractive",
"Bags_Under_Eyes",
"Bald",
"Bangs",
"Big_Lips",
"Big_Nose",
"Black_Hair",
"Blond_Hair",
"Blurry",
"Brown_Hair",
"Bushy_Eyebrows",
"Chubby",
"Double_Chin",
"Eyeglasses",
"Goatee",
"Gray_Hair",
"Heavy_Makeup",
"High_Cheekbones",
"Male",
"Mouth_Slightly_Open",
"Mustache",
"Narrow_Eyes",
"No_Beard",
"Oval_Face",
"Pale_Skin",
"Pointy_Nose",
"Receding_Hairline",
"Rosy_Cheeks",
"Sideburns",
"Smiling",
"Straight_Hair",
"Wavy_Hair",
"Wearing_Earrings",
"Wearing_Hat",
"Wearing_Lipstick",
"Wearing_Necklace",
"Wearing_Necktie",
"Young"]
