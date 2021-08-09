import numpy as np
from itertools import combinations
import torch

def calculate_pi_dimension(input_size, stride):
    nb_patch_x = int(np.ceil(input_size[1]/stride[0]))
    nb_patch_y = int(np.ceil(input_size[2]/stride[1]))
    
    return nb_patch_x, nb_patch_y




def prepare_data(data, target, num_classes=10, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
    one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)

    return data, target, one_hot_target

def get_extended_data(data, Nexpectation):
    shape = data.shape
    data_unsqueezed = data.unsqueeze(0)
    wanted_transform = tuple(np.insert(-np.ones(len(shape),dtype = int),0,Nexpectation))
    
    data_expanded = data_unsqueezed.expand(wanted_transform) # N_expectation, batch_size, channels, size:...

    data_expanded_flatten = data_expanded.flatten(0,1)

    return data_expanded, data_expanded_flatten
    
def prepare_data_augmented(data, target, index=None, num_classes=10, Nexpectation = 1, use_cuda = False):
    if index is not None :
        index_expanded = index.unsqueeze(0).expand(Nexpectation, -1).flatten(0,1)
    else :
        index_expanded = None
    
    if use_cuda:
        data =data.cuda()
        if target is not None :
            target = target.cuda()
    if target is not None :
        one_hot_target = torch.nn.functional.one_hot(target,num_classes=num_classes) # batch_size, category
        one_hot_target_expanded = one_hot_target.unsqueeze(0).expand(Nexpectation,-1,-1) #N_expectations, batch_size, category
    else :
        one_hot_target = None
        one_hot_target_expanded = None
    
    shape = data.shape
    data_unsqueezed = data.unsqueeze(0)
    wanted_transform = tuple(np.insert(-np.ones(len(shape),dtype = int),0,Nexpectation))

     
    data_expanded = data_unsqueezed.expand(wanted_transform) # N_expectation, batch_size, channels, size:...
    data_expanded_flatten = data_expanded.flatten(0,1)

    wanted_shape_flatten = data_expanded_flatten.shape

    return data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded


def print_dic(epoch, batch_idx, dic, dataset):
    percent = 100. * batch_idx / len(dataset.train_loader)
    to_print = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(epoch, batch_idx * dataset.batch_size_train, len(dataset.train_loader.dataset), percent)
    for key in dic.keys():
        to_print += "{}: {:.5f} \t".format(key, dic[key])
    print(to_print)

def save_dic_helper(total_dic, dic):
    if len(total_dic.keys())==0:
        for element in dic.keys():
            total_dic[element] = [dic[element]]
    else:
        for element in dic.keys():
            total_dic[element].append(dic[element])

    return total_dic



def get_all_z(dim):
    # list_z = []
    output = np.zeros((2**dim, dim))
    number = 0
    for nb_pos in range(dim+1):
        combinations_index = combinations(range(dim), r = nb_pos)
        for combination in combinations_index :
            for index in combination :
                output[number,index]=1
            number+=1
    # output = torch.tensor(output, dtype=torch.int64).unsqueeze(1).expand(-1, batch_size, -1)
    output = torch.tensor(output, dtype=torch.float)

    return output
