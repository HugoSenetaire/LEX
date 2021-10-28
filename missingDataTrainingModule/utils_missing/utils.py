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

def on_cuda(data, target = None, index = None,):
    if target is not None :
        target = target.cuda()
    if index is not None :
        index = index.cuda()
    data = data.cuda()
    return data, target, index

def get_one_hot(target, num_classes = 10):
    if target is not None :
        one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)
    else :
        one_hot_target = None

    return one_hot_target

def extend_input(input, Nexpectation = 1, nb_imputation = 1):
    shape = input.shape
    if nb_imputation is not None :
        wanted_shape = torch.Size((nb_imputation, Nexpectation)) + shape
        input_expanded = input.unsqueeze(0).unsqueeze(0).expand(wanted_shape)
    else :
        wanted_shape = torch.Size((Nexpectation,)) + shape
        input_expanded = input.unsqueeze(0).expand(wanted_shape)


    return input_expanded
    

    
def prepare_data_augmented(data, target = None, index=None, one_hot_target = None, Nexpectation = 1, nb_imputation = None):
    if index is not None :
        index_expanded = extend_input(index, Nexpectation, nb_imputation)
    else :
        index_expanded = None

    if one_hot_target is not None :
        one_hot_target_expanded = extend_input(one_hot_target, Nexpectation, nb_imputation)
    else :
        one_hot_target_expanded = None

    
    
    if target is not None :
        target_expanded = extend_input(target, Nexpectation, nb_imputation)
    else :
        target_expanded = None
    data_expanded = extend_input(data, Nexpectation, nb_imputation)
     


    return data_expanded, target_expanded, index_expanded, one_hot_target_expanded


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
