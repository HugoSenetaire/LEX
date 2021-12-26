
import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")
from missingDataTrainingModule import *
from datasets import *
from default_parameter import *


from torch.distributions import *
from torch.optim import *
from functools import partial
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler = get_default()
    final_path, trainer_var, loader, dic_list = experiment(args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = True)


    ## Interpretation
    # trainer_var.eval()

    # data, target= next(iter(loader.test_loader))
    # data = data[:20]
    # target = target[:20]

    # sampling_distribution_test = args_test["sampling_distribution_test"]
    # if sampling_distribution_test is RelaxedBernoulli:
    #     current_sampling_test = partial(RelaxedBernoulli,args_test["temperature_test"])
    # else :
    #     current_sampling_test = copy.deepcopy(sampling_distribution_test)
        

    # pred = trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader)
    # image_output, _ = trainer_var._get_pi(data.cuda())
    # image_output = trainer_var.classification_module.patch_creation(image_output)
    # image_output = image_output.detach().cpu().numpy()
    # save_interpretation(final_path, image_output, data, target, suffix = "direct_destruction",
    #                     y_hat= torch.exp(pred).detach().cpu().numpy(),
    #                     class_names= [str(i) for i in range(10)])


    # pred = trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader)
    # pi_list, loss_reg, z, p_z = trainer_var._destructive_test(data.cuda(), sampling_distribution_test, 1)
    # z = trainer_var.classification_module.patch_creation(z)
    # destructed_image, _ = trainer_var.classification_module.imputation.impute(data.cuda(), z)
    # if mask :
    #     mask_index = destructed_image.shape[1]//2
    #     destructed_image = destructed_image[:,:mask_index,:,:]
    # destructed_image = destructed_image.detach().cpu().numpy()
    # save_interpretation(final_path, destructed_image, data, target, suffix = "destructed_image", 
    #                     y_hat= torch.exp(pred).detach().cpu().numpy(),
    #                     class_names= [str(i) for i in range(10)])