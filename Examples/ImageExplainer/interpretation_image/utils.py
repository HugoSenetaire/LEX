import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import copy

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

def imputation_image(trainer, loader, final_path, nb_samples_image_per_category = 3, nb_imputation = 3,):
    trainer.eval()
    data, target, index= next(iter(loader.test_loader))
    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
    total_image = len(data)
    wanted_shape = data[0].shape
    transpose_set = (1, 2, 0)
    if wanted_shape[0] == 1:
        wanted_shape = (wanted_shape[1], wanted_shape[2])
        cmap = 'gray'
    if wanted_shape[0] >1 :
        wanted_shape = (wanted_shape[1], wanted_shape[2], wanted_shape[0])
        cmap = 'viridis'

    if trainer.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    classification_module = trainer.classification_module
    selection_module = trainer.selection_module
    distribution_module = trainer.distribution_module
    classification_module.imputation.nb_imputation_mc_test = nb_imputation
    classification_module.imputation.nb_imputation_iwae_test = 1



    
    log_pi_list,_ = selection_module(data)

    pz = distribution_module(torch.exp(log_pi_list,))
    z = distribution_module.sample((1,))
    z = trainer.reshape(z)
    data_imputed = classification_module.get_imputation(data, z)


    data_imputed = data_imputed.cpu().detach().numpy().reshape(nb_imputation, total_image, *wanted_shape)
    data = data.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    folder_path = os.path.join(final_path, "imputation_from_sample")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    for k in range(total_image):
        for l in range(nb_imputation):
            x_imputed = data_imputed[l][k].transpose(transpose_set).reshape(wanted_shape)
            x_original = data[k].transpose(transpose_set).reshape(wanted_shape)

            current_z = z[k,0].reshape(wanted_shape[:2])
            fig, axs = plt.subplots(1,3, figsize=(15,5))
            axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
            axs[1].imshow(current_z, cmap='gray', interpolation='none',)
            axs[2].imshow(x_imputed, cmap=cmap, interpolation='none',)
            plt.savefig(os.path.join(folder_path, f"{k}_target_{target[k]}_imputation_{l}.png"))
            plt.close(fig)


    
def interpretation_sampled(trainer, loader, final_path, nb_samples_image_per_category = 3):
    trainer.eval()
    data, target, index= next(iter(loader.test_loader))

    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
    wanted_shape = data[0].shape

    transpose_set = (1, 2, 0)
    if wanted_shape[0] == 1:
        wanted_shape = (wanted_shape[1], wanted_shape[2])
        cmap = 'gray'
    if wanted_shape[0] >1 :
        wanted_shape = (wanted_shape[1], wanted_shape[2], wanted_shape[0])
        cmap = 'viridis'


    
    classification_module = trainer.classification_module
    selection_module = trainer.selection_module
    distribution_module = trainer.distribution_module

    if trainer.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    
    log_pi_list,_ = selection_module(data)

    pz = distribution_module(torch.exp(log_pi_list))
    z = distribution_module.sample((1,))
    z = trainer.reshape(z)
    pred, _ = classification_module(data, z)

    data = data.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    pred = torch.exp(pred).cpu().detach().numpy()

    folder_path = os.path.join(final_path, "output_sample")
    for k in range(len(data)):
        target_path = os.path.join(folder_path, f"target_{target[k]}")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        x_original = data[k].transpose(transpose_set).reshape(wanted_shape)
        current_z = z[k, 0].reshape(wanted_shape[:2])
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
        axs[1].imshow(current_z, cmap='gray', interpolation='none',)
        axs[2].bar(np.arange(len(pred[k])), pred[k])
        plt.savefig(os.path.join(target_path, f"{k}.png"))
        plt.close(fig)

    
def image_f1_score(trainer, loader, final_path, nb_samples_image_per_category = 3):
    trainer.eval()
    data, target, index= next(iter(loader.test_loader))
    output_category = loader.dataset.get_dim_output()
    indexes = torch.cat([torch.where(target==num)[0][:nb_samples_image_per_category] for num in range(output_category)])
    data = data[indexes]
    target = target[indexes]
    if not hasattr(loader.dataset, "optimal_S_test"):
        return None

    quadrant = loader.dataset.optimal_S_test[index]
    wanted_shape = data[0].shape
    transpose_set = (0,1,2)
    if wanted_shape[0] == 1:
        wanted_shape = (wanted_shape[1], wanted_shape[2])
        cmap = 'gray'
    if wanted_shape[0] >1 :
        wanted_shape = (wanted_shape[1], wanted_shape[2], wanted_shape[0])
        cmap = 'viridis'
    
    classification_module = trainer.classification_module
    selection_module = trainer.selection_module
    distribution_module = trainer.distribution_module

    if trainer.use_cuda:
        data, target = data.cuda(), target.cuda()
    
    
    log_pi_list,_ = selection_module(data)
    log_pi_list = log_pi_list.cpu().detach().numpy()
    argmax_pi_list = np.round(np.exp(log_pi_list)).astype(int)


   

    data = data.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    folder_path = os.path.join(final_path, "output_quadrant_from_pi_list")
    for k in range(len(data)):
        target_path = os.path.join(folder_path, f"target_{target[k]}")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        x_original = data[k].transpose(transpose_set).reshape(wanted_shape)
        current_quadrant = quadrant[k].reshape(wanted_shape[:2])
        current_pi_list = argmax_pi_list[k].reshape(wanted_shape[:2]) 
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(x_original, cmap=cmap, interpolation='none',)
        axs[1].imshow(current_pi_list, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
        axs[2].imshow(current_quadrant, cmap='gray', interpolation='none', vmin = 0., vmax = 1.0)
        f1_score = metrics.f1_score(current_quadrant.flatten(), current_pi_list.flatten(),)
        plt.title(f"F1_score {f1_score:.5f}")
        plt.savefig(os.path.join(target_path, f"{k}.png"))
        plt.close(fig)
    


def accuracy_output(trainer, loader, final_path, batch_size = 100):
    trainer.eval()
    if not hasattr(loader.dataset, "optimal_S_test"):
        return None
    selection_module = trainer.selection_module
    f1_score_avg = 0
    complete_size = 0
    

    for aux in iter(loader.test_loader):
        data, target, index=aux
        wanted_shape = data[0].shape
        optimal_S_test = loader.dataset.optimal_S_test[index]

        if trainer.use_cuda:
            data, target, = data.cuda(), target.cuda()

        batch_size = data.shape[0]
    
        log_pi_list,_ = selection_module(data)
        pi_list = torch.exp(trainer.reshape(log_pi_list))[:,0]
    
        data = data.cpu().detach().numpy()
        pi_list = pi_list.cpu().detach().numpy().reshape(batch_size, -1)
        optimal_S_test = optimal_S_test.cpu().detach().numpy().reshape(batch_size, -1)
        argmax_pi_list = np.round(pi_list).astype(int)

        accuracy = np.sum(np.abs(optimal_S_test - pi_list).reshape(-1))
        for pi,current_quadrant in zip(argmax_pi_list, optimal_S_test):
            f1_score_avg += metrics.f1_score(current_quadrant, pi,)
        target = target.cpu().detach().numpy()
        complete_size += batch_size

    mean_accuracy = accuracy/complete_size/np.prod(wanted_shape)
    f1_score_avg = f1_score_avg/complete_size
    with open(os.path.join(final_path, "accuracy_output.txt"), "w") as f:
        f.write(f"mean accuracy: {mean_accuracy}\n")
        f.write(f"f1 score: {f1_score_avg}")
        
       
def complete_analysis_image(trainer, loader, final_path, batch_size = 100, nb_samples_image_per_category = 3):
    # Interpretation:
    imputation_image(trainer, loader, final_path)
    interpretation_sampled(trainer, loader, final_path)
    accuracy_output(trainer, loader, final_path, batch_size = batch_size,)
    image_f1_score(trainer, loader, final_path, nb_samples_image_per_category = nb_samples_image_per_category)

    aux_final_path = os.path.join(final_path, "at_best_iter")
    if not os.path.exists(aux_final_path):
        os.makedirs(aux_final_path)

    if os.path.exists(os.path.join(final_path, "classification_module.pt")):
        trainer.load_best_iter_dict(final_path)

    imputation_image(trainer, loader, aux_final_path)
    interpretation_sampled(trainer, loader, aux_final_path)
    accuracy_output(trainer, loader, aux_final_path, batch_size = batch_size)
    image_f1_score(trainer, loader, aux_final_path, nb_samples_image_per_category = nb_samples_image_per_category)

