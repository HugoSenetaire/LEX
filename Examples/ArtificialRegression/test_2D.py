import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")

from missingDataTrainingModule.classification_training import ordinaryTraining, EVAL_X
from missingDataTrainingModule.selection_training import selectionTraining
from missingDataTrainingModule.interpretation_training import SELECTION_BASED_CLASSIFICATION, REALX
from missingDataTrainingModule import PytorchDistributionUtils, utils_reshape, Classification, Selection, main_launcher
from missingDataTrainingModule.PytorchDistributionUtils.gradientestimator import AllCombination, REINFORCE, PathWise, REBAR
from interpretation_regression import *
from missingDataTrainingModule.Classification.imputation import ConstantImputation, DatasetBasedImputation, ModuleImputation, MultipleConstantImputation
from datasets import *
from default_parameter import *

from torch.distributions import *
from torch.optim import *
from functools import partial

if __name__ == '__main__' :
    args_output, args_dataset, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler = get_default()


    args_dataset["dataset"] = LinearDataset
    args_dataset["give_index"] = True

    
    dataset, loader = get_dataset(args_dataset)

    path_weights = os.path.join(args_output["path"], "2DDataset")


    args_train["post_hoc"] = False
    args_train["argmax_post_hoc"] = False
    args_train["post_hoc_guidance"] = None
    args_train["nb_epoch"] = 100 # Training the complete model
 
    args_train["fix_classifier_parameters"] = False
    args_train["nb_sample_z_train_monte_carlo"] = 1 # Number K in the IWAE-similar loss 
    args_train["nb_sample_z_train_IWAE"] = 1
    args_classification["nb_imputation"] = 1



    args_complete_trainer["complete_trainer"] = REALX
    args_complete_trainer["monte_carlo_gradient_estimator"] = PytorchDistributionUtils.gradientestimator.REBAR
    args_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.REBARBernoulli
    args_distribution_module["distribution"] = Bernoulli
    args_distribution_module["distribution_relaxed"] = RelaxedBernoulli
    args_classification["imputation"] = MultipleConstantImputation
    args_classification["cste_imputation"] = -1
    # args_compiler["optim_classification"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) #Learning rate for classification module
    # args_compiler["optim_selection"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) # Learning rate for selection module

    # args_complete_trainer["complete_trainer"] = SELECTION_BASED_CLASSIFICATION
    # args_complete_trainer["monte_carlo_gradient_estimator"] = AllCombination # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    # args_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.DistributionModule
    # args_distribution_module["distribution"] = Bernoulli
    # args_classification["imputation"] = Classification.imputation.DatasetBasedImputation


    # args_complete_trainer["complete_trainer"] = SELECTION_BASED_CLASSIFICATION
    # args_complete_trainer["monte_carlo_gradient_estimator"] = PytorchDistributionUtils.gradientestimator.REBAR # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    # args_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.REBARBernoulli_STE
    # args_distribution_module["distribution"] = Bernoulli
    # args_classification["imputation"] = Classification.imputation.DatasetBasedImputation

    # args_complete_trainer["complete_trainer"] = SELECTION_BASED_CLASSIFICATION
    # args_complete_trainer["monte_carlo_gradient_estimator"] = PytorchDistributionUtils.gradientestimator.PathWise # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    # args_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.DistributionWithTemperatureParameter
    # args_distribution_module["distribution"] = RelaxedBernoulli_thresholded_STE
    # args_classification["imputation"] = Classification.imputation.DatasetBasedImputation



    args_selection["rate"] = 0.
    args_selection["loss_regularization"] = "L1"
    args_selection["batched"] = False

    name_experiment = "DatasetBasedImputation"

    path_global = os.path.join(path_weights, name_experiment)


    list_lambda = [0.1, 0.5, 1., 2., 5., 10.]

    for lambda_reg in list_lambda :

        args_selection["lambda_reg"] = lambda_reg


        aux_string = f"{str(args_complete_trainer['complete_trainer']).split('.')[-1][:-2]}_multiple_cste_{str(args_complete_trainer['monte_carlo_gradient_estimator']).split('.')[-1][:-2]}_{args_train['fix_classifier_parameters']}_{args_selection['loss_regularization']}_{args_selection['batched']}"
        current_path = os.path.join(path_global, aux_string)
        current_path = os.path.join(current_path, f"{lambda_reg}_IWAESAMPLE_{args_train['nb_sample_z_train_IWAE'] }")
        args_output["path"] = current_path

        dataset, loader = get_dataset(args_dataset)
        final_path, trainer_var, loader, dic_list = main_launcher.experiment(dataset, loader, args_output, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = False)
        if loader.dataset.nb_dim ==2:
            plot_selector_output(trainer_var.selection_module, loader.dataset, args_output["path"])
            plot_selector_output(trainer_var.selection_module, loader.dataset, args_output["path"], train_data=True)
            plot_selector_output(trainer_var.selection_module, loader.dataset, args_output["path"], interpretation= True)
            plot_selector_output(trainer_var.selection_module, loader.dataset, args_output["path"], interpretation= True, train_data=True)
            plot_complete_model_output(trainer_var, loader.dataset, Bernoulli, args_output["path"])
            plot_complete_model_output(trainer_var, loader.dataset, Bernoulli, args_output["path"], train_data=True)
        


    

    