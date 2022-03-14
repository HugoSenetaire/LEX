
import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")
from missingDataTrainingModule import *
from datasets import *
from interpretation_image import *
from default_parameter import *


from torch.distributions import *
from torch.optim import *
from functools import partial
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, args_classification_distribution_module = get_default()
    dataset, loader = get_dataset(args_dataset)

    origin_path = args_output["path"]
    name_experiment = "ToeplitzRealXDataset"

    list_classifier = {"ClassifierREALX" : Classification.classification_network.RealXClassifier, "ClassifierLvl2": Classification.classification_network.ClassifierLVL2}
    list_selector = {"SelectorRealX": RealXSelector,  "SelectorLinear": Selection.selective_network.SelectorLinear, } 
    path_global = os.path.join(origin_path, name_experiment)

    list_nb_component_imputation = [2, 20, 50, 100]
    list_lambda = [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    list_mc_iwae = [(1,1,1,1), (100, 1, 1, 1), (1, 100, 1, 1), (1, 1, 100, 1), (1, 1, 1, 100),]
    count = 1
    for nb_components in list_nb_component_imputation:
        path_weights = "dataset_{}_{}_components.pkl".format("FASHIONMNIST_and_MNIST", nb_components)
        path_weights = os.path.join(path_global, path_weights)
        if not os.path.exists(path_weights):
            train_gmm(dataset.data_train.numpy().reshape(-1, np.prod(args_selection["input_size_selector"])), nb_components, path_weights)
        args_classification["module_imputation"] = GaussianMixtureImputation(path_weights) # Path to the weights of the network to use for post processing)
        for classifier_name, selector_name in zip(list_classifier, list_selector):
            classifier = list_classifier[classifier_name]
            selector = list_selector[selector_name]
            for lambda_reg in list_lambda :
                for mc_mask, iwae_mask, mc_sample, iwae_sample in list_mc_iwae :
                    for loss in ["MSE", "NLL"]:
                        args_train["loss_function"] = loss # NLL, MSE
                        args_train["nb_sample_z_train_monte_carlo"] = mc_mask # Number of samples for monte carlo gradient estimator
                        args_train["nb_sample_z_train_IWAE"] = iwae_mask # Number K in the IWAE-similar loss 
                        args_classification["nb_imputation_mc"] = mc_sample
                        args_classification["nb_imputation_iwae"] = iwae_sample
                        args_selection["lambda_reg"] = lambda_reg
                        args_classification["classifier"] = classifier
                        args_selection["selector"] = selector

                        dic = {}
                        aux_string = f"{str(args_complete_trainer['complete_trainer']).split('.')[-1][:-2]}_{str(args_complete_trainer['monte_carlo_gradient_estimator']).split('.')[-1][:-2]}_{args_train['fix_classifier_parameters']}_{args_selection['loss_regularization']}_{args_selection['batched']}"
                        current_path = os.path.join(path_global, aux_string)
                        current_path = os.path.join(current_path, f"{str(classifier_name)}_{str(selector_name)}_{lambda_reg}_{count}")
                        count +=1
                        args_output["path"] = current_path
                        final_path, trainer_var, loader, dic_list = main_launcher.experiment(dataset, loader, args_output, args_classification,
                                                                            args_selection, args_distribution_module, args_complete_trainer,
                                                                            args_train, args_test, args_compiler, args_classification_distribution_module, name_modification = False)
                        final_path, trainer, loader, dic_list = experiment(dataset, loader, args_output, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = True)

                        # Interpretation:
                        imputation_image(trainer, loader, final_path)
                        interpretation_sampled(trainer, loader, final_path)
                        accuracy_output(trainer, loader, final_path, batch_size = 100)
                        image_f1_score(trainer, loader, final_path, nb_samples_image = 20)
                        