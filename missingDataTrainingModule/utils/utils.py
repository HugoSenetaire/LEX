import numpy as np
from itertools import combinations
import torch
import os

def calculate_pi_dimension(input_size, stride):
    nb_patch_x = int(np.ceil(input_size[1]/stride[0]))
    nb_patch_y = int(np.ceil(input_size[2]/stride[1]))
    
    return nb_patch_x, nb_patch_y


def parse_batch(data):
    if len(data)==3 :
        input, target, index = data
    else :
        input, target = data
        index = None

    return input, target, index
    
def get_item(tensor):
    if tensor is not None :
        return tensor.item()
    else :
        return None

def prepare_data(data, target, index = None, num_classes=10, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
        if index is not None :
            index = index.cuda()
    if target is not None and num_classes > 1:
        one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)
    else :
        one_hot_target = None

    return data, target, one_hot_target, index



def get_extended_data(data, nb_sample_z):
    shape = data.shape
    data_unsqueezed = data.unsqueeze(0)
    wanted_transform = tuple(np.insert(-np.ones(len(shape),dtype = int),0,nb_sample_z))
    
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
    if target is not None and num_classes > 1:
        one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)
    else :
        one_hot_target = None

    return one_hot_target


def extend_input(input, mc_part = 1, iwae_part = 1,):
    shape = input.shape
    reshape_shape = torch.Size((1,)) + torch.Size((shape[0],)) + torch.Size((1,)) + shape[1:]
    wanted_shape = torch.Size((mc_part,)) + torch.Size((shape[0],)) + torch.Size((iwae_part,)) + shape[1:]
    input_expanded = input.reshape(reshape_shape).expand(wanted_shape)
    return input_expanded
     

def sampling_augmentation(data, target = None, index=None, one_hot_target = None, mc_part = 1, iwae_part = 1,):
    if index is not None :
        index_expanded = extend_input(index, mc_part = mc_part, iwae_part = iwae_part,)
    else :
        index_expanded = None

    if one_hot_target is not None :
        one_hot_target_expanded = extend_input(one_hot_target, mc_part = mc_part, iwae_part = iwae_part,)
    else :
        one_hot_target_expanded = None

    if target is not None :
        target_expanded = extend_input(target, mc_part = mc_part, iwae_part = iwae_part,)
    else :
        target_expanded = None

    data_expanded = extend_input(data, mc_part = mc_part, iwae_part = iwae_part,)
     
    return data_expanded, target_expanded, index_expanded, one_hot_target_expanded


def print_dic(epoch, batch_idx, dic, loader):
    percent = 100. * batch_idx / len(loader.train_loader)
    to_print = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(epoch, batch_idx * loader.batch_size_train, len(loader.train_loader.dataset), percent)
    for key in dic.keys():
        if dic[key] is not None :
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
    output = np.zeros((2**dim, dim))
    number = 0
    for nb_pos in range(dim+1):
        combinations_index = combinations(range(dim), r = nb_pos)
        for combination in combinations_index :
            for index in combination :
                output[number,index]=1
            number+=1
    output = torch.tensor(output, dtype=torch.float)

    return output


def dic_evaluation(accuracy, mse, neg_likelihood, suffix = "", mse_loss_prod = None, confusion_matrix = None,):
    dic = {}
    
    dic["accuracy" + suffix] = accuracy
    dic["mse" + suffix] = mse
    dic["neg_likelihood" + suffix] = neg_likelihood
    if mse_loss_prod is not None :
        dic["mse_loss_prod" + suffix] = mse_loss_prod
    if confusion_matrix is not None :
        dic["confusion_matrix" + suffix] = confusion_matrix
    return dic


def save_model(final_path, prediction_module, selection_module, distribution_module, baseline, suffix = ""):
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    path = os.path.join(final_path, f"prediction_module{suffix}.pt")
    torch.save(prediction_module.state_dict(), path)

    path = os.path.join(final_path, f"selection_module{suffix}.pt")
    torch.save(selection_module.state_dict(), path)

    path = os.path.join(final_path, f"distribution_module{suffix}.pt")
    torch.save(distribution_module.state_dict(), path)

    if baseline is not None:
        path = os.path.join(final_path, f"baseline{suffix}.pt")
        torch.save(baseline.state_dict(), path)


import torch
import gc

def memory_manager(print=True, save_path = None,):
    if save_path is not None:
        txt = "Memory used: {}\n".format(torch.cuda.memory_allocated())
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
                if save_path is not None:
                    txt += "Type {}, Size: {} Device: {} \n".format(type(obj), obj.size(), obj.device)
        except:
            pass
    with open(save_path, "a") as f:
        f.write(txt)