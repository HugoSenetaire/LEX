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

def imputation_image(interpretable_module, loader, final_path, nb_samples_image_per_category = 3, nb_imputation = 3,):
    interpretable_module.eval()
    data, target, index= next(iter(loader.test_loader))
    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
    total_image = len(data)

    wanted_shape = loader.dataset.get_dim_input()
    if wanted_shape[0] == 1 :
        transpose_set = None
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2])
        cmap = 'gray'
    elif wanted_shape[0] >1 :
        transpose_set = (1,2,0)
        wanted_shape_transpose = (wanted_shape[1], wanted_shape[2], wanted_shape[0])
        cmap = 'viridis'

    if interpretable_module.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    prediction_module = interpretable_module.prediction_module
    selection_module = interpretable_module.selection_module
    distribution_module = interpretable_module.distribution_module
    prediction_module.imputation.nb_imputation_mc_test = nb_imputation
    prediction_module.imputation.nb_imputation_iwae_test = 1

    
    log_pi_list,_ = selection_module(data)


    pz = distribution_module(torch.exp(log_pi_list,))
    z = distribution_module.sample((1,))
    z = interpretable_module.reshape(z)


    data_imputed = prediction_module.get_imputation(data, z)
    data_imputed = data_imputed.cpu().detach().numpy().reshape(nb_imputation, total_image, *wanted_shape)
    data = data.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    folder_path = os.path.join(final_path, "imputation_from_sample")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    for k in range(total_image):
        for l in range(nb_imputation):
            x_imputed = data_imputed[l][k]
            x_original = data[k]
            if transpose_set is not None :
                x_imputed = x_imputed.transpose(transpose_set)
                x_original = x_original.transpose(transpose_set)
            x_imputed = x_imputed.reshape(wanted_shape_transpose)
            x_original = x_original.reshape(wanted_shape_transpose)

            current_z = z[k,0].reshape(wanted_shape_transpose[:2])
            fig, axs = plt.subplots(1,3, figsize=(15,5))
            axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
            axs[0].set_title(f"Original image")
            axs[1].imshow(current_z, cmap='gray', interpolation='none', vmin = 0., vmax = 1.)
            axs[1].set_title(f"One Sampled mask")
            axs[2].imshow(x_imputed, cmap=cmap, interpolation='none',)
            axs[2].set_title(f"Imputed image")
            plt.axis('off')
            plt.savefig(os.path.join(folder_path, f"{k}_target_{target[k]}_imputation_{l}.png"))
            plt.close(fig)


    
def interpretation_sampled(interpretable_module, loader, final_path, nb_samples_image_per_category = 3):
    interpretable_module.eval()
    data, target, index= next(iter(loader.test_loader))

    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
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

    if interpretable_module.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    
    log_pi_list,_ = selection_module(data)

    pz = distribution_module(torch.exp(log_pi_list))
    z = distribution_module.sample((100,))
    z = interpretable_module.reshape(z)
    pi_list = z.mean(dim=0)
    pred, _ = prediction_module(data, z[0,...])

    data = data.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    pred = torch.exp(pred).cpu().detach().numpy()
    pi_list = pi_list.cpu().detach().numpy()

    folder_path = os.path.join(final_path, "output_sample")
    for k in range(len(data)):
        target_path = os.path.join(folder_path, f"target_{target[k]}")
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        x_original = data[k]
        if transpose_set is not None :
            x_original = x_original.transpose(transpose_set)
        x_original = x_original.reshape(wanted_shape_transpose)

        current_z = z[k, 0].reshape(wanted_shape_transpose[:2])
        current_pi_list = pi_list[k, 0].reshape(wanted_shape_transpose[:2])
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

    
def image_f1_score(interpretable_module, loader, final_path, nb_samples_image_per_category = 3):
    interpretable_module.eval()
    data, target, _ = next(iter(loader.test_loader))
    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
    if not hasattr(loader.dataset, "optimal_S_test"):
        return None

    quadrant = loader.dataset.optimal_S_test[indexes]
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

    if interpretable_module.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    
    log_pi_list,_ = selection_module(data)
    pz = distribution_module(torch.exp(log_pi_list))
    z = distribution_module.sample((100,))
    z = interpretable_module.reshape(z)
    pi_list = z.mean(dim=0)


    data = data.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    folder_path = os.path.join(final_path, "output_quadrant_from_pi_list")
    for k in range(len(data)):
        target_path = os.path.join(folder_path, f"target_{target[k]}")
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        x_original = data[k]
        if transpose_set is not None :
            x_original = x_original.transpose(transpose_set)
        x_original = x_original.reshape(wanted_shape_transpose)
        
        current_quadrant = quadrant[k].reshape(wanted_shape_transpose[:2])
        current_pi_list = pi_list[k].reshape(wanted_shape_transpose[:2]) 
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
        axs[0].set_title(f"Original image")
        axs[1].imshow(current_pi_list, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
        axs[1].set_title(f"Average z sampled")
        axs[2].imshow(current_quadrant, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
        axs[2].set_title(f"True selection")
        plt.axis("off")
        f1_score = metrics.f1_score(current_quadrant.flatten(), current_pi_list.flatten(),)
        plt.title(f"F1_score {f1_score:.5f}")
        plt.savefig(os.path.join(target_path, f"{k}.png"))
        plt.close(fig)
    
       
def complete_analysis_image(interpretable_module, loader, trainer, args, batch_size = 100, nb_samples_image_per_category = 3,):
    interpretable_module.eval()
    imputation_image(interpretable_module, loader, args.args_output.path)
    interpretation_sampled(interpretable_module, loader, args.args_output.path)
    image_f1_score(interpretable_module, loader, args.args_output.path, nb_samples_image_per_category = nb_samples_image_per_category)

    dic = eval_selection_sample(interpretable_module, loader,)

    for key in list(dic.keys()) :
        dic["sampled_" +key] = dic[key]

    dic.update(test_epoch(interpretable_module, None, loader, args, liste_mc = [(1,1,1,1),], trainer = trainer,))
    return dic