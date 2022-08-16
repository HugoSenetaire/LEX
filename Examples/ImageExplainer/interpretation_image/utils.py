import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import copy
import json

from missingDataTrainingModule.EvaluationUtils import eval_selection_sample, test_epoch

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


def save_imputation(x_original, target, x_imputed, z, final_path, k, l, wanted_shape_transpose, transpose_set = None, cmap = 'gray'):
    folder_path = os.path.join(final_path, "imputation_from_sample")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if transpose_set is not None :
        x_imputed = x_imputed.transpose(transpose_set)
        x_original = x_original.transpose(transpose_set)
    x_imputed = x_imputed.reshape(wanted_shape_transpose)
    x_original = x_original.reshape(wanted_shape_transpose)

    current_z = z[0].reshape(wanted_shape_transpose[:2])
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[1].imshow(current_z, cmap='gray', interpolation='none', vmin = 0., vmax = 1.)
    axs[1].set_title(f"One Sampled Mask")
    axs[2].imshow(x_imputed, cmap=cmap, interpolation='none',)
    axs[2].set_title(f"Imputed image")
    plt.axis('off')

    plt.savefig(os.path.join(folder_path, f"{k}_target_{target}_imputation_{l}.png"))
    plt.close(fig)


def save_sampling_mask(x_original, z,  pi_list, target, pred, final_path, k, wanted_shape_transpose, transpose_set = None, cmap = 'gray', prefix = "",):
    folder_path = os.path.join(final_path, "output_sample")
    target_path = os.path.join(folder_path, f"target_{target}")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if transpose_set is not None :
        x_original = x_original.transpose(transpose_set)
    x_original = x_original.reshape(wanted_shape_transpose)

    current_z = z.reshape(wanted_shape_transpose[:2])
    current_pi_list = pi_list.reshape(wanted_shape_transpose[:2])
    
    fig, axs = plt.subplots(1,4, figsize=(20,5))
    axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[1].imshow(current_z, cmap='gray', interpolation='none', vmin = 0., vmax = 1.)
    axs[1].set_title(f"One sample")
    axs[2].imshow(current_pi_list, cmap='gray', interpolation='none', vmin = 0., vmax = 1.)
    axs[2].set_title(f"Sample averaged")
    axs[3].bar(np.arange(len(pred[k])), pred[k])
    axs[3].set_title(f"Prediction using the one sample")
    plt.axis("off")
    plt.savefig(os.path.join(target_path, f"{k}.png"))
    plt.close(fig)



def save_f1score(data, target, quadrant, pi_list, final_path, k, wanted_shape_transpose, transpose_set = None, cmap = 'gray'):
    folder_path = os.path.join(final_path, "f1_score")

    target_path = os.path.join(folder_path, f"target_{target}")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    x_original = data
    if transpose_set is not None :
        x_original = x_original.transpose(transpose_set)
    x_original = x_original.reshape(wanted_shape_transpose)
    
    current_quadrant = quadrant.reshape(wanted_shape_transpose[:2])
    current_pi_list = pi_list.reshape(wanted_shape_transpose[:2]) 
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
    axs[0].set_title(f"Original image")
    axs[1].imshow(current_pi_list, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
    axs[1].set_title(f"Average z sampled")
    axs[2].imshow(current_quadrant, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
    axs[2].set_title(f"True selection")
    plt.axis("off")
    try :
        f1_score = metrics.f1_score(current_quadrant.flatten(), current_pi_list.flatten(),)
        plt.title(f"F1_score {f1_score:.5f}")
    except ValueError:
        f1_score = None
    
    plt.savefig(os.path.join(target_path, f"{k}.png"))
    plt.close(fig)
    
    
def interpretation_image(interpretable_module, loader, final_path, nb_samples_image_per_category = 3, nb_imputation = 3,):
    interpretable_module.eval()
    data, target, index= next(iter(loader.test_loader))


    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target =  target[indexes]
    if hasattr(loader.dataset, "optimal_S_test"):
        quadrant = loader.dataset.optimal_S_test[indexes]
    total_image = len(data)

    # data = loader.dataset.data_test[indexes]
    # target = loader.dataset.target_test[indexes]
    wanted_shape = loader.dataset.get_dim_input()
    if wanted_shape[0] == 1 :
        transpose_set = None
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2])
        cmap = 'gray'
    elif wanted_shape[0] >1 :
        transpose_set = (1, 2, 0)
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2], wanted_shape[0])
        cmap = 'viridis'

    


    prediction_module = interpretable_module.prediction_module
    selection_module = interpretable_module.selection_module
    distribution_module = interpretable_module.distribution_module

    prediction_module.imputation.nb_imputation_mc_test = nb_imputation
    prediction_module.imputation.nb_imputation_iwae_test = 1

    if next(interpretable_module.parameters()).is_cuda:
        data, target = data.cuda(), target.cuda()
    
    
    log_pi_list,_ = selection_module(data)
    pi_list = torch.exp(log_pi_list)
    pi_list = interpretable_module.reshape(pi_list[None, :, None])

    

    pz = distribution_module(torch.exp(log_pi_list))
    z = distribution_module.sample((100,))
    pi_list_sampled = z.mean(dim=0)[None,]
    pi_list_sampled = interpretable_module.reshape(pi_list_sampled)
    

    z = distribution_module.sample((1,))
    z = interpretable_module.reshape(z)

    data_imputed = prediction_module.get_imputation(data, z)
    data_imputed = data_imputed.reshape(nb_imputation, total_image, *wanted_shape)

    prediction_module.imputation.nb_imputation_mc_test = 1
    pred, _ = prediction_module(data, z)

    data = data.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    pred = torch.exp(pred).cpu().detach().numpy()
    pi_list_sampled = pi_list_sampled.cpu().detach().numpy()
    pi_list = pi_list.cpu().detach().numpy()
    data_imputed = data_imputed.cpu().detach().numpy()
    quadrant = quadrant.cpu().detach().numpy() if hasattr(loader.dataset, "optimal_S_test") else None

    # print("data shape", data.shape)
    # print("z shape", z.shape)
    # print("target shape", target.shape)
    # print("pred shape", pred.shape)
    # print("pi_list_sampled shape", pi_list_sampled.shape)
    # print("pi_list shape", pi_list.shape)
    # print("data_imputed shape", data_imputed.shape)
    # print("quadrant shape", quadrant.shape) if hasattr(loader.dataset, "optimal_S_test") else None


    for k in range(len(data)):
        save_sampling_mask(data[k], z[k, 0], pi_list_sampled[k, 0], target[k], pred, final_path, k, wanted_shape_transpose, transpose_set, cmap,)
        if hasattr(loader.dataset, "optimal_S_test"):
            save_f1score(data[k], target[k], quadrant[k], pi_list[k], final_path, k, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, cmap = cmap,)
        for l in range(nb_imputation):
            save_imputation(data[k], target[k], data_imputed[l][k], z[k], final_path, k, l, wanted_shape_transpose = wanted_shape_transpose, transpose_set = transpose_set, cmap = cmap)



       
def complete_analysis_image(interpretable_module, loader, trainer, args, batch_size = 100, nb_samples_image_per_category = 1, nb_imputation = 1,):
    print("Starting analysis")
    interpretable_module.eval()
    interpretation_image(interpretable_module, loader, args.args_output.path, nb_samples_image_per_category = nb_samples_image_per_category, nb_imputation = nb_imputation,)

    interpretable_module.prediction_module.imputation.nb_imputation_mc_test = args.args_classification.nb_imputation_mc_test
    interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = args.args_classification.nb_imputation_iwae_test

    dic = eval_selection_sample(interpretable_module, loader,)

    for key in list(dic.keys()) :
        dic["sampled_" +key] = dic[key]

    dic.update(test_epoch(interpretable_module, "Analysis", loader, args, liste_mc = [(1,1,1,1),], trainer = trainer,))
    return dic