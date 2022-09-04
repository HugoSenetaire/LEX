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

def prepare_data(data, target, index = None, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
        if index is not None :
            index = index.cuda()

    return data, target, index



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

def class_or_reg(output_dim):
    if type(output_dim) == int :
        if output_dim == 1 :
            problem_type = "regression"
        else :
            problem_type = "classification"
    elif len(output_dim) == 1 :
        if np.prod(output_dim) == 1 :
            problem_type = "regression"
        else :
            problem_type = "classification"
    else :
        problem_type = "regression"
    return problem_type

def extend_input(input, mc_part = 1, iwae_part = 1,):
    if input is None :
        return None
    shape = input.shape
    reshape_shape = torch.Size((1,)) + torch.Size((shape[0],)) + torch.Size((1,)) + shape[1:]
    wanted_shape = torch.Size((mc_part,)) + torch.Size((shape[0],)) + torch.Size((iwae_part,)) + shape[1:]
    input_expanded = input.reshape(reshape_shape).expand(wanted_shape)
    return input_expanded
     

def sampling_augmentation(data, target = None, index=None, mc_part = 1, iwae_part = 1,):
    if index is not None :
        index_expanded = extend_input(index, mc_part = mc_part, iwae_part = iwae_part,)
    else :
        index_expanded = None


    if target is not None :
        target_expanded = extend_input(target, mc_part = mc_part, iwae_part = iwae_part,)
    else :
        target_expanded = None

    data_expanded = extend_input(data, mc_part = mc_part, iwae_part = iwae_part,)
     
    return data_expanded, target_expanded, index_expanded


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


def save_model(final_path, interpretable_module, suffix = ""):
    print("Saving model at {}".format(final_path))
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    try :
        path = os.path.join(final_path, f"prediction_module{suffix}.pt")
        torch.save(interpretable_module.prediction_module.state_dict(), path)
    except AttributeError:
        print("No prediction module to save")

    try :
        path = os.path.join(final_path, f"selection_module{suffix}.pt")
        torch.save(interpretable_module.selection_module.state_dict(), path)
    except AttributeError:
        print("No selection module to save")
    
    try :
        path = os.path.join(final_path, f"distribution_module{suffix}.pt")
        torch.save(interpretable_module.distribution_module.state_dict(), path)
    except AttributeError:
        print("No distribution module to save")
    
    try :
        path = os.path.join(final_path, f"baseline{suffix}.pt")
        torch.save(interpretable_module.baseline.state_dict(), path)
    except AttributeError:
        print("No baseline to save")

def load_model(final_path, prediction_module = None, selection_module = None, distribution_module = None, baseline = None, suffix = "_best"):
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    if prediction_module is not None :
        path = os.path.join(final_path, f"prediction_module{suffix}.pt")
        prediction_module.load_state_dict(torch.load(path))

    if selection_module is not None :
        path = os.path.join(final_path, f"selection_module{suffix}.pt")
        selection_module.load_state_dict(torch.load(path))

    if distribution_module is not None :
        path = os.path.join(final_path, f"distribution_module{suffix}.pt")
        distribution_module.load_state_dict(torch.load(path))

    if baseline is not None:
        path = os.path.join(final_path, f"baseline{suffix}.pt")
        baseline.load_state_dict(torch.load(path))

    return prediction_module, selection_module, distribution_module, baseline


import torch
import gc

def memory_manager(save_path = None,):
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
    print("Saved at {}".format(save_path))
    print(os.path.exists(save_path))