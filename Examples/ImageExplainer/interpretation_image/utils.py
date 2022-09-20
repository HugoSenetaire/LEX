import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import copy
import json

from missingDataTrainingModule.EvaluationUtils import eval_selection_sample, test_epoch, get_sel_pred, eval_selection



def show_interpretation(sample, data, target, shape = (1,28,28)):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,3,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.subplot(1,3,3)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=sample_reshaped[k].min().item(), vmax=sample_reshaped[k].max().item())
        plt.show()

def handle_multiple_channel(x, wanted_shape, transpose_set = None):
    """
    Given a single input image, get the channel at the right place if necessary
    """

    if len(x.shape) > 2 and x.shape[0] > 1:
        if transpose_set is not None:
            x = x.transpose(transpose_set)
            x = x.reshape(wanted_shape)
            cmap = "viridis"
        else:
            raise ValueError("The input image has multiple channels but no transpose set is provided")
    else :
        x = x.reshape(wanted_shape[:2])
        cmap = "gray"
    
    return x, cmap


def save_imputation(x_original, target, x_imputed, z, final_path, k, l, wanted_shape_transpose, transpose_set = None, ):
    folder_path = os.path.join(final_path, "imputation_from_sample")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    x_original, cmap_x = handle_multiple_channel(x_original, wanted_shape_transpose, transpose_set = transpose_set)
    current_z, cmap_z = handle_multiple_channel(z, wanted_shape_transpose, transpose_set = transpose_set)
    x_imputed, cmap_imputed = handle_multiple_channel(x_imputed, wanted_shape_transpose, transpose_set = transpose_set)

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(x_original, cmap=cmap_x, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[0].axis("off")
    axs[1].imshow(current_z, cmap=cmap_z, interpolation='none', vmin = 0., vmax = 1.)
    axs[1].set_title(f"One Sampled Mask")
    axs[1].axis("off")
    axs[2].imshow(x_imputed, cmap=cmap_imputed, interpolation='none',)
    axs[2].set_title(f"Imputed image")
    axs[2].axis("off")
    plt.axis('off')
    plt.savefig(os.path.join(folder_path, f"{k}_target_{target}_imputation_{l}.png"))
    plt.close(fig)


def save_sampling_mask(x_original, z,  pi_list, target, pred, final_path, k, wanted_shape_transpose, transpose_set = None, prefix = "",):
    folder_path = os.path.join(final_path, "output_sample")
    target_path = os.path.join(folder_path, f"target_{target}")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    x_original, cmap_x = handle_multiple_channel(x_original, wanted_shape_transpose, transpose_set = transpose_set)
    current_z, cmap_z = handle_multiple_channel(z, wanted_shape_transpose, transpose_set = transpose_set)
    current_pi_list, cmap_pi_list = handle_multiple_channel(pi_list, wanted_shape_transpose, transpose_set = transpose_set)


    fig, axs = plt.subplots(1,4, figsize=(20,5))
    axs[0].imshow(x_original, cmap=cmap_x, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[0].axis("off")
    axs[1].imshow(current_z, cmap=cmap_z, interpolation='none', vmin = 0., vmax = 1.)
    axs[1].set_title(f"One sample")
    axs[1].axis("off")
    axs[2].imshow(current_pi_list, cmap=cmap_pi_list, interpolation='none', vmin = 0., vmax = 1.)
    axs[2].set_title(f"Sample averaged")
    axs[2].axis("off")
    axs[3].bar(np.arange(len(pred[k])), pred[k])
    axs[3].set_title(f"Prediction using the one sample")
    plt.savefig(os.path.join(target_path, f"{k}.png"))
    plt.close(fig)



def save_f1score(data, target, quadrant, pi_list, final_path, k, wanted_shape_transpose, transpose_set = None, pi_list_estimation = "Undefined"):
    folder_path = os.path.join(final_path, f"f1_score_{pi_list_estimation}")

    target_path = os.path.join(folder_path, f"target_{target}")
    if not os.path.exists(target_path):
        os.makedirs(target_path)




    x_original, cmap_x = handle_multiple_channel(data, wanted_shape_transpose, transpose_set = transpose_set)
    current_quadrant, cmap_quadrant = handle_multiple_channel(quadrant, wanted_shape_transpose, transpose_set = transpose_set)
    current_pi_list, cmap_pi_list = handle_multiple_channel(pi_list, wanted_shape_transpose, transpose_set = transpose_set)


    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(x_original, cmap=cmap_x, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[0].axis("off")
    axs[1].imshow(current_pi_list, cmap=cmap_quadrant, interpolation='none', vmin = 0., vmax = 1.0)
    axs[1].set_title(f"Pi list estimated with {pi_list_estimation}")
    axs[1].axis("off")
    axs[2].imshow(current_quadrant, cmap=cmap_pi_list, interpolation='none', vmin = 0., vmax = 1.0)
    axs[2].set_title(f"True selection")
    axs[2].axis("off")
    plt.axis("off")
    try :
        f1_score = metrics.f1_score(current_quadrant.flatten(), current_pi_list.flatten(),)
        plt.title(f"F1_score {f1_score:.5f}")
    except ValueError:
        f1_score = None
    plt.savefig(os.path.join(target_path, f"{k}.png"))
    plt.close(fig)
    
def get_enough_data_per_target(loader, nb_samples_image_per_category):
    output_category = loader.dataset.get_dim_output()
    nb_samples_image_per_category_index = [nb_samples_image_per_category for i in range(output_category)]
    total_data = []
    total_target = []
    total_indexes = []
    for index in range(len(loader.test_loader.dataset)):
        aux = loader.test_loader.dataset.__getitem__(index)
        
        data, target = aux[0], aux[1]
        if nb_samples_image_per_category_index[target] > 0:
            nb_samples_image_per_category_index[target] -= 1
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index, dtype = torch.int64)
            total_data += [data]
            total_indexes += [index]
            total_target += [target]
        if sum(nb_samples_image_per_category_index) <= 0:
            break
    total_target = torch.stack(total_target)
    total_data = torch.stack(total_data)
    total_indexes = torch.stack(total_indexes)
    if hasattr(loader.dataset, "get_true_selection"):
        quadrant = loader.dataset.get_true_selection(total_indexes)
    else :
        quadrant = None
    
    return total_indexes, total_data, total_target, quadrant
    
def interpretation_image(interpretable_module, loader, final_path, nb_samples_image_per_category = 1, nb_imputation = 3, rate = None):
    interpretable_module.eval()

    indexes, data, target, quadrant = get_enough_data_per_target(loader, nb_samples_image_per_category)
    total_image = len(data)

    shape_input = interpretable_module.prediction_module.input_size
    if shape_input[0] == 1:
        channel_handling = False
    else :
        channel_handling = True
    dim = np.prod(shape_input[1:])

    wanted_shape = loader.dataset.get_dim_input()
    if wanted_shape[0] == 1 :
        transpose_set = None
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2])
    elif wanted_shape[0] >1 :
        transpose_set = (1, 2, 0)
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2], wanted_shape[0])

    


    prediction_module = interpretable_module.prediction_module
    selection_module = interpretable_module.selection_module
    distribution_module = interpretable_module.distribution_module

    prediction_module.imputation.nb_imputation_mc_test = nb_imputation
    prediction_module.imputation.nb_imputation_iwae_test = 1

    if next(interpretable_module.parameters()).is_cuda:
        data, target = data.cuda(), target.cuda()
    
    with torch.no_grad():
        log_pi_list,_ = selection_module(data)
        pi_list = torch.exp(log_pi_list)
        pi_list_selected = get_sel_pred(interpretable_module, pi_list, rate = rate)
        pz = distribution_module(torch.exp(log_pi_list))


        pi_list_selected = interpretable_module.reshape(pi_list_selected)
        pi_list = interpretable_module.reshape(pi_list)

        z = distribution_module.sample((100,))
        pi_list_sampled = z.mean(dim=0)
        pi_list_sampled = interpretable_module.reshape(pi_list_sampled)
        

        z = distribution_module.sample((1,)).flatten(0,1)
        z = interpretable_module.reshape(z)

        data_imputed = prediction_module.get_imputation(data, z)
        data_imputed = data_imputed.reshape(nb_imputation, total_image, *wanted_shape)

        prediction_module.imputation.nb_imputation_mc_test = 1
        pred, _ = prediction_module(data, z)

    data = data.cpu().detach().numpy()
    data_imputed = data_imputed.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    pred = torch.exp(pred).cpu().detach().numpy()

    if channel_handling :
        pi_list_sampled = pi_list_sampled[:,0]
        pi_list = pi_list[:,0]
        pi_list_selected = pi_list_selected[:,0]
        z = z[:,0]

    pi_list_sampled = pi_list_sampled.cpu().detach().numpy()
    pi_list = pi_list.cpu().detach().numpy()
    pi_list_selected = pi_list_selected.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    
    if hasattr(loader.dataset, "get_true_selection"):
        quadrant = quadrant.to(torch.float32).cpu().detach().numpy()
        if channel_handling :
            quadrant = quadrant[:,0]

    for k in range(len(data)):
        save_sampling_mask(data[k], z[k,], pi_list_sampled[k,], target[k], pred, final_path, k, wanted_shape_transpose, transpose_set,)
        if hasattr(loader.dataset, "get_true_selection"):
            save_f1score(data[k], target[k], quadrant[k], pi_list[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "straight_pi_list")
            save_f1score(data[k], target[k], quadrant[k], pi_list_selected[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "selected_pi_list")
            save_f1score(data[k], target[k], quadrant[k], pi_list_sampled[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "sampled_pi_list")
        else :
            save_f1score(data[k], target[k], np.zeros_like(pi_list[k]), pi_list[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "straight_pi_list")
            save_f1score(data[k], target[k], np.zeros_like(pi_list[k]), pi_list_selected[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "selected_pi_list")
            save_f1score(data[k], target[k], np.zeros_like(pi_list[k]), pi_list_sampled[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, pi_list_estimation = "sampled_pi_list")

        for l in range(nb_imputation):
            save_imputation(data[k], target[k], data_imputed[l][k], z[k], final_path, k, l, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, )



       
def complete_analysis_image(interpretable_module, loader, trainer, args, batch_size = 100, nb_samples_image_per_category = 1, nb_imputation = 1,):
    print("Starting analysis")
    interpretable_module.eval()
    interpretation_image(interpretable_module, loader, args.args_output.path, nb_samples_image_per_category = nb_samples_image_per_category, nb_imputation = nb_imputation, rate = args.args_selection.rate)

    interpretable_module.prediction_module.imputation.nb_imputation_mc_test = args.args_classification.nb_imputation_mc_test
    interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = args.args_classification.nb_imputation_iwae_test

    dic = eval_selection_sample(interpretable_module, loader,)

    for key in list(dic.keys()) :
        dic["sampled_" +key] = dic[key]

    dic.update(eval_selection(interpretable_module, loader,args))
    if trainer is not None :
        dic.update(test_epoch(interpretable_module, "Analysis", loader, args, liste_mc = [(1,1,1,1),], trainer = trainer,))
    
    return dic